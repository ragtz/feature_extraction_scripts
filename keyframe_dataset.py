
import numpy as np
import itertools
import pickle

class KeyframeDataset(object):
    def __init__(self, data=None):
        self.data = data
        if not data is None:
            self.pids = self._get_pids()
            self.tasks = self._get_tasks()
            self.n = len(self.data['pid'])

    def _get_keys(self, key):
        return list(set(self.data[key]))

    def _get_pids(self):
        return self._get_keys('pid')

    def _get_tasks(self):
        return self._get_keys('task')

    def _get_kwvals(self, key, **kwargs):
        vals = []

        if key in kwargs:
            if type(kwargs[key]) is list:
                vals.extend(kwargs[key])
            else:
                vals.append(kwargs[key])

        if not vals:
            vals = self._get_keys(key)

        return vals

    def _make_np(self, d):
        for k in d:
            d[k] = np.array(d[k])

    def _check_nonempty(self):
        if self.data is None:
            raise Exception('No data in keyframe dataset')

    def _check_subset_size(self, l, m):
        if m > len(l)-1:
            raise Exception('Subset size must be less than or equal to the number of elements')

    def _get_empty_dataset(self):
        dataset = {'pid': [],
                   'task': [],
                   'demo_id': [],
                   'kf_idx': [],
                   'kf': [],
                   'label': []}
        return dataset

    def _get_dataset(self, **kwargs):
        pids = self._get_kwvals('pid', **kwargs)
        tasks = self._get_kwvals('task', **kwargs)
        dataset = self._get_empty_dataset()

        for i in range(self.n):
            pid = self.data['pid'][i]
            task = self.data['task'][i]
            demo_id = self.data['demo_id'][i]
            kf_idx = self.data['kf_idx'][i]
            kf = self.data['kf'][i]
            label = self.data['label'][i]

            if pid in pids and task in tasks:
                self._add_data(dataset, pid, task, demo_id, kf_idx, kf, label)

        self._make_np(dataset)
       
        return dataset 

    def _add_data(self, dataset, pid, task, demo_id, kf_idx, kf, label):
        dataset['pid'].append(pid)
        dataset['task'].append(task)
        dataset['demo_id'].append(demo_id)
        dataset['kf_idx'].append(kf_idx)
        dataset['kf'].append(kf)
        dataset['label'].append(label)
    
    def _get_feature_label(self, **kwargs):
        data = self._get_dataset(**kwargs)
        return (data['kf'], data['label'])

    def _iter_subsets(self, l, m):
        return itertools.combinations(l, m)

    def load(self, data_file):
        self.data = pickle.load(open(data_file))
        self.pids = self._get_pids()
        self.tasks = self._get_tasks()
        self.n = len(self.data['pid'])

    def get_num_pids(self):
        return len(self.pids)

    def get_num_tasks(self):
        return len(self.tasks)

    def get_keyframe_dataset(self, **kwargs):
        self._check_nonempty()
        dataset = self._get_dataset(**kwargs)
        return KeyframeDataset(dataset)

    def get_keyframes(self):
        return self.data['kf']

    def get_labels(self):
        return self.data['label']

    def iter_train_test(self, m):
        self._check_nonempty()
        self._check_subset_size(self.pids, m)

        for pid_train_subset in self._iter_subsets(self.pids, m):
            pid_train_subset = list(pid_train_subset)
            pid_test_subset = list(set(self.pids) - set(pid_train_subset))

            yield (pid_train_subset, self._get_feature_label(pid=pid_train_subset)), (pid_test_subset, self._get_feature_label(pid=pid_test_subset))

    def task_iter_train_test(self, m):
        self._check_nonempty()
        self._check_subset_size(self.tasks, m)

        for task_train_subset in self._iter_subsets(self.tasks, m):
            task_train_subset = list(task_train_subset)
            task_test_subset = list(set(self.tasks) - set(task_train_subset))

            yield (task_train_subset, self._get_feature_label(task=task_train_subset)), (task_test_subset, self._get_feature_label(task=task_test_subset))

