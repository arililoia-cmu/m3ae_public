import threading
from io import BytesIO
from queue import Queue

# import gcsfs
# import h5py
import numpy as np
# import skimage.io
import torch
import torchvision
import transformers
from ml_collections import ConfigDict
from PIL import Image
from skimage.color import gray2rgb, rgba2rgb
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torchvision
from torchvision import transforms


class ImageTextDataset(torch.utils.data.Dataset):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = ""

        config.start_index = 0
        config.max_length = int(1e9)
        config.random_start = False

        config.image_only = False
        config.tokenize = True
        config.tokenizer = "bert-base-uncased"
        config.tokenizer_max_length = 64

        config.transform_type = "pretrain"
        config.image_size = 256

        config.image_normalization = 'cc12m'
        config.custom_image_mean = ''
        config.custom_image_std = ''

        config.random_drop_text = 0.0
        config.deterministic_drop_text = 0.0

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, start_offset_ratio=None):
        self.config = self.get_default_config(config)
        assert self.config.path != ""

        if self.config.image_normalization == 'imagenet':
            self.image_mean = (0.485, 0.456, 0.406)
            self.image_std = (0.229, 0.224, 0.225)
        elif self.config.image_normalization == 'cc12m':
            self.image_mean = (0.5762, 0.5503, 0.5213)
            self.image_std = (0.3207, 0.3169, 0.3307)
        elif self.config.image_normalization == 'none':
            self.image_mean = (0.0, 0.0, 0.0)
            self.image_std = (1.0, 1.0, 1.0)
        elif self.config.image_normalization == 'custom':
            self.image_mean = tuple(float(x) for x in self.config.custom_image_mean.split('-'))
            self.image_std = tuple(float(x) for x in self.config.custom_image_std.split('-'))
            assert len(self.image_mean) == len(self.image_std) == 3
        else:
            raise ValueError('Unsupported image normalization mode!')

        if self.config.path.startswith("gs://"):
            # Loading from GCS
            self.h5_file = h5py.File(
                gcsfs.GCSFileSystem().open(self.config.path, cache_type="block"), "r"
            )
        else:
            self.h5_file = h5py.File(self.config.path, "r")

        if self.config.transform_type == "pretrain":
            # Use Kaiming's simple pretrain processing
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        self.config.image_size,
                        scale=(0.2, 1.0),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.image_mean, std=self.image_std),
                ]
            )
        elif self.config.transform_type == "finetune":
            # Use Kaiming's finetune processing
            self.transform = create_transform(
                input_size=self.config.image_size,
                is_training=True,
                color_jitter=True,
                auto_augment=None,
                interpolation="bicubic",
                re_prob=0,
                re_mode=0,
                re_count="const",
                mean=self.image_mean,
                std=self.image_std,
            )
        elif self.config.transform_type == "test":
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        self.config.image_size,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.CenterCrop(self.config.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.image_mean, std=self.image_std),
                ]
            )
        elif self.config.transform_type == 'resize_only':
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        self.config.image_size,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.CenterCrop(self.config.image_size),
                    transforms.ToTensor(),
                ]
            )
        else:
            raise ValueError("Unsupported transform_type!")

        if self.config.tokenize:
            self.tokenizer = transformers.BertTokenizer.from_pretrained(
                self.config.tokenizer
            )

        if self.config.random_start:
            # Bypass numpy random seed
            self.random_start_offset = np.random.default_rng().choice(len(self))
        elif start_offset_ratio is not None:
            self.random_start_offset = int(len(self) * start_offset_ratio) % len(self)
        else:
            self.random_start_offset = 0

    def __getstate__(self):
        return self.config, self.random_start_offset

    def __setstate__(self, state):
        config, random_start_offset = state
        self.__init__(config)
        self.random_start_offset = random_start_offset

    def __len__(self):
        return min(
            self.h5_file["jpg"].shape[0] - self.config.start_index,
            self.config.max_length,
        )

    def process_index(self, index):
        index = (index + self.random_start_offset) % len(self)
        return index + self.config.start_index

    def drop_text(self, raw_index):
        deterministic_drop = float(raw_index % 100) / 100. < self.config.deterministic_drop_text
        random_drop = np.random.rand() < self.config.random_drop_text
        return deterministic_drop or random_drop

    def __getitem__(self, raw_index):

        # get the audio
        index = self.process_index(raw_index)
        with BytesIO(self.h5_file["jpg"][index]) as fin:
            image = skimage.io.imread(fin)

        if len(image.shape) == 2:
            image = gray2rgb(image)
        elif image.shape[-1] == 4:
            image = rgba2rgb(image)

        image = (
            self.transform(Image.fromarray(np.uint8(image))).permute(1, 2, 0).numpy()
        )
        image = image.astype(np.float32)
        if self.config.image_only:
            return image

        with BytesIO(self.h5_file["caption"][index]) as fin:
            caption = fin.read().decode("utf-8")

        if not self.config.tokenize:
            return image, caption

        if len(caption) == 0 or self.drop_text(raw_index):
            tokenized_caption = np.zeros(self.config.tokenizer_max_length, dtype=np.int32)
            padding_mask = np.ones(self.config.tokenizer_max_length, dtype=np.float32)
            return image, tokenized_caption, padding_mask

        encoded_caption = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.config.tokenizer_max_length,
            return_tensors="np",
            add_special_tokens=False,
        )

        if encoded_caption["input_ids"][0].size == 0:  # Empty token
            tokenized_caption = np.zeros(self.config.tokenizer_max_length, dtype=np.int32)
            padding_mask = np.ones(self.config.tokenizer_max_length, dtype=np.float32)
        else:
            tokenized_caption = encoded_caption["input_ids"][0]
            padding_mask = 1.0 - encoded_caption["attention_mask"][0].astype(np.float32)

        return image, tokenized_caption, padding_mask

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def text_length(self):
        return self.config.tokenizer_max_length


# audioset dataset taken from:
# https://raw.githubusercontent.com/facebookresearch/AudioMAE/bd60e29651285f80d32a6405082835ad26e6f19f/dataset.py
class AudiosetDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf, label_csv=None, use_fbank=False, fbank_dir=None, roll_mag_aug=False, load_video=False, mode='train'):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)
        self.use_fbank = use_fbank
        self.fbank_dir = fbank_dir

        self.data = data_json['data']
        self.audio_conf = audio_conf
        print('---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        if 'multilabel' in self.audio_conf.keys():
            self.multilabel = self.audio_conf['multilabel']
        else:
            self.multilabel = False
        print(f'multilabel: {self.multilabel}')
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        print('using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup')
        print('using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        print('Dataset: {}, mean {:.3f} and std {:.3f}'.format(self.dataset, self.norm_mean, self.norm_std))
        self.noise = self.audio_conf.get('noise')
        if self.noise == True:
            print('now use noise augmentation')
        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        self.roll_mag_aug=roll_mag_aug
        print(f'number of classes: {self.label_num}')
        print(f'size of dataset {self.__len__()}')


    def _roll_mag_aug(self, waveform):
        waveform=waveform.numpy()
        idx=np.random.randint(len(waveform))
        rolled_waveform=np.roll(waveform,idx)
        mag = np.random.beta(10, 10) + 0.5
        return torch.Tensor(rolled_waveform*mag)

    def _wav2fbank(self, filename, filename2=None):
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
            if self.roll_mag_aug:
                waveform = self._roll_mag_aug(waveform)
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if self.roll_mag_aug:
                waveform1 = self._roll_mag_aug(waveform1)
                waveform2 = self._roll_mag_aug(waveform2)

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()
        # 498 128, 998, 128
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        # 512
        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        if filename2 == None:
            return fbank, 0
        else:
            return fbank, mix_lambda


    def _fbank(self, filename, filename2=None):
        if filename2 == None:
            fn1 = os.path.join(self.fbank_dir, os.path.basename(filename).replace('.wav','.npy'))
            fbank = np.load(fn1)
            return torch.from_numpy(fbank), 0
        else:
            fn1 = os.path.join(self.fbank_dir, os.path.basename(filename).replace('.wav','.npy'))
            fn2 = os.path.join(self.fbank_dir, os.path.basename(filename2).replace('.wav','.npy'))
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)
            fbank = mix_lambda * np.load(fn1) + (1-mix_lambda) * np.load(fn2)  
            return torch.from_numpy(fbank), mix_lambda

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        # do mix-up for this sample (controlled by the given mixup rate)
        if random.random() < self.mixup: # for audio_exp, when using mixup, assume multilabel
            datum = self.data[index]
            # find another sample to mix, also do balance sampling
            # sample the other sample from the multinomial distribution, will make the performance worse
            # mix_sample_idx = np.random.choice(len(self.data), p=self.sample_weight_file)
            # sample the other sample from the uniform distribution
            mix_sample_idx = random.randint(0, len(self.data)-1)
            mix_datum = self.data[mix_sample_idx]

            # get the mixed fbank
            if not self.use_fbank:
                fbank, mix_lambda = self._wav2fbank(datum['wav'], mix_datum['wav'])
            else:
                fbank, mix_lambda = self._fbank(datum['wav'], mix_datum['wav'])
            # initialize the label
            label_indices = np.zeros(self.label_num)
            # add sample 1 labels
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += mix_lambda
            # add sample 2 labels
            for label_str in mix_datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += 1.0-mix_lambda
            label_indices = torch.FloatTensor(label_indices)
        # if not do mixup
        else:
            datum = self.data[index]
            label_indices = np.zeros(self.label_num)
            if not self.use_fbank:
                fbank, mix_lambda = self._wav2fbank(datum['wav'])
            else:
                fbank, mix_lambda = self._fbank(datum['wav'])
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] = 1.0

            if self.multilabel:
                label_indices = torch.FloatTensor(label_indices)
            else:
                # remark : for ft cross-ent
                label_indices = int(self.index_dict[label_str])
        # SpecAug for training (not for eval)
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = fbank.transpose(0,1).unsqueeze(0) # 1, 128, 1024 (...,freq,time)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank) # (..., freq, time)
        fbank = torch.transpose(fbank.squeeze(), 0, 1) # time, freq
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        if self.noise == True: # default is false, true for spc
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)
        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return fbank.unsqueeze(0), label_indices, datum['wav']

    def __len__(self):
        return len(self.data)



# class ImageNetDataset(torch.utils.data.Dataset):
#     @staticmethod
#     def get_default_config(updates=None):
#         config = ConfigDict()
#         config.path = ""
#         config.partition = "train"
#         config.image_only = False

#         config.start_index = 0
#         config.max_length = int(1e9)
#         config.random_start = False

#         config.image_normalization = 'imagenet'
#         config.transform_type = "pretrain"
#         config.image_size = 256

#         config.autoaug = "rand-m9-mstd0.5-inc1"

#         if updates is not None:
#             config.update(ConfigDict(updates).copy_and_resolve_references())
#         return config

#     def __init__(self, config, start_offset_ratio=None):
#         self.config = self.get_default_config(config)
#         assert self.config.path != ""

#         if self.config.path.startswith("gs://"):
#             # Loading from GCS
#             self.h5_file = h5py.File(
#                 gcsfs.GCSFileSystem().open(self.config.path, cache_type="block"), "r"
#             )
#         else:
#             self.h5_file = h5py.File(self.config.path, "r")

#         if self.config.image_normalization == 'imagenet':
#             self.image_mean = (0.485, 0.456, 0.406)
#             self.image_std = (0.229, 0.224, 0.225)
#         elif self.config.image_normalization == 'cc12m':
#             self.image_mean = (0.5762, 0.5503, 0.5213)
#             self.image_std = (0.3207, 0.3169, 0.3307)
#         elif self.config.image_normalization == 'none':
#             self.image_mean = (0.0, 0.0, 0.0)
#             self.image_std = (1.0, 1.0, 1.0)
#         elif self.config.image_normalization == 'custom':
#             self.image_mean = tuple(float(x) for x in self.config.custom_image_mean.split('-'))
#             self.image_std = tuple(float(x) for x in self.config.custom_image_std.split('-'))
#             assert len(self.image_mean) == len(self.image_std) == 3
#         else:
#             raise ValueError('Unsupported image normalization mode!')

#         if self.config.transform_type == "pretrain":
#             # Use Kaiming's simple pretrain processing
#             self.transform = transforms.Compose(
#                 [
#                     transforms.RandomResizedCrop(
#                         self.config.image_size,
#                         scale=(0.2, 1.0),
#                         interpolation=transforms.InterpolationMode.BICUBIC,
#                     ),
#                     transforms.RandomHorizontalFlip(),
#                     transforms.ToTensor(),
#                     transforms.Normalize(
#                         mean=self.image_mean, std=self.image_std
#                     ),
#                 ]
#             )
#         elif self.config.transform_type == "finetune":
#             # Use Kaiming's finetune processing
#             self.transform = create_transform(
#                 input_size=self.config.image_size,
#                 is_training=True,
#                 color_jitter=True,
#                 auto_augment=self.config.autoaug,
#                 interpolation="bicubic",
#                 re_prob=0,
#                 re_mode=0,
#                 re_count="const",
#                 mean=self.image_mean,
#                 std=self.image_std,
#             )
#         elif self.config.transform_type == "plain_finetune":
#             # Use supervised training processing of ViT from "Better plain ViT baselines for ImageNet-1k" https://arxiv.org/abs/2205.01580
#             self.transform = transforms.Compose(
#                 [
#                     transforms.RandomResizedCrop(
#                         self.config.image_size,
#                         interpolation=transforms.InterpolationMode.BICUBIC,
#                     ),
#                     transforms.RandomHorizontalFlip(),
#                     transforms.ToTensor(),
#                     transforms.Normalize(
#                         mean=self.image_mean, std=self.image_std
#                     ),
#                 ]
#             )
#         elif self.config.transform_type == "linear_prob":
#             self.transform = transforms.Compose(
#                 [
#                     transforms.RandomResizedCrop(
#                         self.config.image_size,
#                         interpolation=transforms.InterpolationMode.BICUBIC,
#                     ),
#                     transforms.RandomHorizontalFlip(),
#                     transforms.ToTensor(),
#                     transforms.Normalize(
#                         mean=self.image_mean, std=self.image_std
#                     ),
#                 ]
#             )
#         elif self.config.transform_type == "test":
#             self.transform = transforms.Compose(
#                 [
#                     transforms.Resize(
#                         self.config.image_size,
#                         interpolation=transforms.InterpolationMode.BICUBIC,
#                     ),
#                     transforms.CenterCrop(self.config.image_size),
#                     transforms.ToTensor(),
#                     transforms.Normalize(
#                         mean=self.image_mean, std=self.image_std
#                     ),
#                 ]
#             )
#         else:
#             raise ValueError("Unsupported transform_type!")

#         if self.config.random_start:
#             # Bypass numpy random seed
#             self.random_start_offset = np.random.default_rng().choice(len(self))
#         elif start_offset_ratio is not None:
#             self.random_start_offset = int(len(self) * start_offset_ratio) % len(self)
#         else:
#             self.random_start_offset = 0

#     def __getstate__(self):
#         return self.config, self.random_start_offset

#     def __setstate__(self, state):
#         config, random_start_offset = state
#         self.__init__(config)
#         self.random_start_offset = random_start_offset

#     def __len__(self):
#         return min(
#             self.h5_file["{}_jpg".format(self.config.partition)].shape[0]
#             - self.config.start_index,
#             self.config.max_length,
#         )

#     def process_index(self, index):
#         index = (index + self.random_start_offset) % len(self)
#         return index + self.config.start_index

#     def __getitem__(self, index):
#         index = self.process_index(index)
#         with BytesIO(
#             self.h5_file["{}_jpg".format(self.config.partition)][index]
#         ) as fin:
#             image = skimage.io.imread(fin)

#         if len(image.shape) == 2:
#             image = gray2rgb(image)
#         elif image.shape[-1] == 4:
#             image = rgba2rgb(image)

#         image = (
#             self.transform(Image.fromarray(np.uint8(image))).permute(1, 2, 0).numpy()
#         )
#         image = image.astype(np.float32)

#         if self.config.image_only:
#             return image

#         label = self.h5_file["{}_labels".format(self.config.partition)][index]

#         return image, label

#     def num_classes(self):
#         return 1000


class TextDataset(torch.utils.data.Dataset):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = ""

        config.start_index = 0
        config.max_length = int(1e9)
        config.random_start = True

        config.tokenize = True
        config.tokenizer = "bert-base-uncased"
        config.tokenizer_max_length = 256

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, start_offset_ratio=None):
        self.config = self.get_default_config(config)
        assert self.config.path != ""

        if self.config.path.startswith("gs://"):
            # Loading from GCS
            self.h5_file = h5py.File(
                gcsfs.GCSFileSystem().open(self.config.path, cache_type="block"), "r"
            )
        else:
            self.h5_file = h5py.File(self.config.path, "r")

        if self.config.tokenize:
            self.tokenizer = transformers.BertTokenizer.from_pretrained(
                self.config.tokenizer
            )

        if self.config.random_start:
            # Bypass numpy random seed
            self.random_start_offset = np.random.default_rng().choice(len(self))
        elif start_offset_ratio is not None:
            self.random_start_offset = int(len(self) * start_offset_ratio) % len(self)
        else:
            self.random_start_offset = 0

    def __getstate__(self):
        return self.config, self.random_start_offset

    def __setstate__(self, state):
        config, random_start_offset = state
        self.__init__(config)
        self.random_start_offset = random_start_offset

    def __len__(self):
        return min(
            self.h5_file["text"].shape[0] - self.config.start_index,
            self.config.max_length,
        )

    def process_index(self, index):
        index = (index + self.random_start_offset) % len(self)
        return index + self.config.start_index

    def __getitem__(self, raw_index):
        index = self.process_index(raw_index)

        with BytesIO(self.h5_file["text"][index]) as fin:
            text = fin.read().decode("utf-8")

        if not self.config.tokenize:
            return text

        if len(text) == 0:
            tokenized = np.zeros(self.config.tokenizer_max_length, dtype=np.int32)
            padding_mask = np.ones(self.config.tokenizer_max_length, dtype=np.float32)
            return tokenized, padding_mask

        encoded_text = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.config.tokenizer_max_length,
            return_tensors="np",
            add_special_tokens=False,
        )

        if encoded_text["input_ids"][0].size == 0:  # Empty token
            tokenized_text = np.zeros(self.config.tokenizer_max_length, dtype=np.int32)
            padding_mask = np.ones(self.config.tokenizer_max_length, dtype=np.float32)
        else:
            tokenized_text = encoded_text["input_ids"][0]
            padding_mask = 1.0 - encoded_text["attention_mask"][0].astype(np.float32)

        return tokenized_text, padding_mask

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size
