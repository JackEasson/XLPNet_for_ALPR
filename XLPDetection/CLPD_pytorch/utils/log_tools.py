import os
from pathlib import Path


class TxtStateWriter(object):
    def __init__(self, file_root: str, file_name: str, state_list: list, resume: bool = False):
        assert isinstance(file_root, str) and isinstance(file_name, str) and isinstance(state_list, list) and \
               file_name.endswith('.txt')
        assert all([isinstance(elem, str) for elem in state_list])
        self.state_list = state_list
        self.resume = resume
        # self.filepath = os.path.join(file_root, file_name)
        self.filepath = Path(file_root) / file_name
        self.__create_new_file()

    def __create_new_file(self):
        if os.path.exists(self.filepath) and not self.resume:  # 有文件，无需保留
            os.remove(self.filepath)
            with open(self.filepath, 'w', encoding='utf-8') as f:
                write_infos = ""
                for s in self.state_list:
                    write_infos += "{:^20s}".format(s)
                f.write(write_infos + '\n')
        elif os.path.exists(self.filepath) and self.resume:  # 有文件，需保留
            with open(self.filepath, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                try:
                    assert all([s in first_line for s in self.state_list])
                except AssertionError:
                    with open(self.filepath, 'a', encoding='utf-8') as f2:
                        write_infos = ""
                        for s in self.state_list:
                            write_infos += "{:^20s}".format(s)
                        f2.write(write_infos + '\n')
        else:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                write_infos = ""
                for s in self.state_list:
                    write_infos += "{:^20s}".format(s)
                f.write(write_infos + '\n')

    def add_info(self, info_list: list):
        assert isinstance(info_list, list) and len(info_list) == len(self.state_list)
        with open(self.filepath, 'a', encoding='utf-8') as f:
            write_infos = ""
            for elem in info_list:
                if elem is None:
                    elem = "None"
                elif not isinstance(elem, str):
                    if isinstance(elem, float):
                        elem = "%.6f" % elem
                    else:
                        elem = str(elem)
                write_infos += "{:^20s}".format(elem)
            f.write(write_infos + '\n')


if __name__ == "__main__":
    writer = TxtStateWriter(file_root='.', file_name='2.txt', state_list=['loss1', 'loss2', 'total_loss'], resume=False)
    writer.add_info([0.1, 0.2, None])



