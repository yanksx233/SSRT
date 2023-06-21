import os
import shutil
import logging


"""
    - log_dir/
        - save_dir1/
            - train/
                - args.log
                - train.log
                - epoch.log
                - checkpoint/
            - val/
                - args.log
                - val.log
                - result/
            - test/
                - args.log
                - test.log
                - result/
        - save_dir2/
            ...
        - save_dir3/
            ...
        ...
"""


def get_logger(log_dir, save_dir, mode, saved_args=None):
    if os.path.exists(os.path.join(log_dir, save_dir, mode)):
        if mode != 'train':
            shutil.rmtree(os.path.join(log_dir, save_dir, mode))
            os.makedirs(os.path.join(log_dir, save_dir, mode, 'result'))
    else:
        os.makedirs(os.path.join(log_dir, save_dir, mode, 'checkpoint' if mode == 'train' else 'result'))

    # save args
    if saved_args:
        args_file = open(os.path.join(log_dir, save_dir, mode, 'args.log'), 'w')
        for k, v in vars(saved_args).items():
            args_file.write(k.rjust(20) + '\t' + str(v) + '\n')

    # logger setting
    logger = logging.getLogger(name=mode + 'logger')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s')

    file_handler = logging.FileHandler(os.path.join(log_dir, save_dir, mode, mode + '.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if mode == 'train':
        info_handler = logging.FileHandler(os.path.join(log_dir, save_dir, mode, 'epoch.log'))
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(formatter)
        logger.addHandler(info_handler)

    return logger