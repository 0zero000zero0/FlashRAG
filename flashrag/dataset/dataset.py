import json
import os
import random

import numpy as np


class Item:
    r"""A container class used to store and manipulate a sample within a dataset.
    Information related to this sample during training/inference will be stored in ```self.output```.
    Each attribute of this class can be used like a dict key(also for key in ```self.output```).

    """

    def __init__(self, item_dict):
        self.id = item_dict.get("id", None)
        self.question = item_dict.get("question", None)
        self.golden_answers = item_dict.get("golden_answers", [])
        self.choices = item_dict.get("choices", [])
        self.metadata = item_dict.get("metadata", {})
        self.output = item_dict.get("output", {})

    def update_output(self, key, value):
        r"""Update the output dict and keep a key in self.output can be used as an attribute."""
        if key in ["id", "question", "golden_answers", "output"]:
            raise AttributeError(f"{key} should not be changed")
        else:
            self.output[key] = value

    def update_evaluation_score(self, metric_name, metric_score):
        r"""Update the evaluation score of this sample for a metric."""
        if "metric_score" not in self.output:
            self.output["metric_score"] = {}
        self.output["metric_score"][metric_name] = metric_score

    def __getattr__(self, attr_name):
        if attr_name in [
            "id",
            "question",
            "golden_answers",
            "metadata",
            "output",
            "choices",
        ]:
            return super().__getattribute__(attr_name)
        else:
            output = super().__getattribute__("output")
            if attr_name in output:
                return output[attr_name]
            else:
                raise AttributeError(f"Attribute `{attr_name}` not found")

    def to_dict(self):
        r"""Convert all information within the data sample into a dict. Information generated
        during the inference will be saved into output field.
        """

        def convert_ndarray(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, dict):
                return {k: convert_ndarray(v) for k, v in value.items()}
            if isinstance(value, list):
                return [convert_ndarray(v) for v in value]
            if isinstance(value, tuple):
                return tuple(convert_ndarray(v) for v in value)
            return value

        # Serialize all attributes, including fields added dynamically via upate_item.
        return {k: convert_ndarray(v) for k, v in self.__dict__.items()}

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def upate_item(self, new_dict: dict):
        for k, v in new_dict.items():
            self.__setitem__(k, v)


class Dataset:
    """A container class used to store the whole dataset. Inside the class, each data sample will be stored
    in ```Item``` class.
    The properties of the dataset represent the list of attributes corresponding to each item in the dataset.
    """

    def __init__(
        self,
        config: dict,
        dataset_path=None,
        data=None,
        sample_num=None,
        random_sample=False,
    ):
        self.config = config
        self.dataset_name = config["dataset_name"]
        self.dataset_path = dataset_path

        self.sample_num = sample_num
        self.random_sample = random_sample
        self.pred = []

        if data is None:
            self.data = self._load_data(self.dataset_name, self.dataset_path)
        else:
            self.data = data

    def _load_data(self, dataset_name, dataset_path):
        """Load data from the provided dataset_path or directly download the file(TODO)."""

        if not os.path.exists(dataset_path):
            # TODO: auto download: self._download(dataset_name, dataset_path)
            pass

        data = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                item_dict = json.loads(line)
                item = Item(item_dict)
                data.append(item)
        if self.sample_num is not None:
            if self.random_sample:
                print(f"Random sample {self.sample_num} items in test set.")
                data = random.sample(data, self.sample_num)
            else:
                data = data[: self.sample_num]

        return data

    def update_output(self, key, value_list):
        """Update the overall output field for each sample in the dataset."""

        assert len(self.data) == len(value_list)
        for item, value in zip(self.data, value_list):
            item.update_output(key, value)

    @property
    def question(self):
        return [item.question for item in self.data]

    @property
    def golden_answers(self):
        return [item.golden_answers for item in self.data]

    @property
    def id(self):
        return [item.id for item in self.data]

    @property
    def output(self):
        return [item.output for item in self.data]

    def get_batch_data(self, attr_name: str, batch_size: int):
        """Get an attribute of dataset items in batch."""

        for i in range(0, len(self.data), batch_size):
            batch_items = self.data[i : i + batch_size]
            yield [item[attr_name] for item in batch_items]

    def __getattr__(self, attr_name):
        return [item.__getattr__(attr_name) for item in self.data]

    def get_attr_data(self, attr_name):
        """For the attributes constructed later (not implemented using property),
        obtain a list of this attribute in the entire dataset.
        """
        return [item[attr_name] for item in self.data]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_batch(self, batch_size):
        for i in range(0, len(self.data), batch_size):
            yield self.data[i : i + batch_size]

    def save_to_json(self, save_path):
        """Save the dataset into the json format."""

        def convert_to_float(d):
            return {
                k: (v.item() if isinstance(v, np.generic) else v) for k, v in d.items()
            }

        save_data = [convert_to_float(item.to_dict()) for item in self.data]
        with open(save_path, "w") as f:
            json.dump(save_data, f, indent=4)

    def save_to_jsonlines(self, save_path):
        """save data with jsonlines format"""

        def convert_to_serializable(d):
            """处理 numpy 数据类型以便能够进行 JSON 序列化"""
            serializable_dict = {}
            for k, v in d.items():
                if isinstance(v, np.generic):
                    serializable_dict[k] = v.item()
                elif isinstance(v, dict):
                    serializable_dict[k] = convert_to_serializable(v)
                elif isinstance(v, list):
                    serializable_dict[k] = [
                        (i.item() if isinstance(i, np.generic) else i) for i in v
                    ]
                else:
                    serializable_dict[k] = v
            return serializable_dict

        with open(save_path, "w", encoding="utf-8") as f:
            for item in self.data:
                item_dict = item.to_dict()
                serializable_dict = convert_to_serializable(item_dict)
                f.write(json.dumps(serializable_dict, ensure_ascii=False) + "\n")
