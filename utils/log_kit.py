import logging


class MyLogger:
    def __init__(self, name, log_level=logging.INFO, log_file=None, log_mode='a'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)

        formatter = logging.Formatter('%(levelname)s %(filename)s:%(lineno)s | %(message)s')

        handlers = []
        stream_handler = logging.StreamHandler()
        handlers.append(stream_handler)

        if log_file is not None:
            file_handler = logging.FileHandler(log_file, log_mode)
            handlers.append(file_handler)

        for handler in handlers:
            handler.setFormatter(formatter)
            handler.setLevel(log_level)
            self.logger.addHandler(handler)


my_logger = MyLogger('mvu', log_level=logging.INFO, log_file='log.txt')
LOGGER = my_logger.logger

# if __name__ == '__main__':
#     # logger = MyLogger('mvu', log_level=logging.WARNING, log_file='log.txt')
#     # logger.log_parameters({'param1': 'value1', 'param2': 'value2'})
#     # logger.logger.warning('This is a warning message')
#     # logger.logger.error('This is an error message')
#     LOGGER = my_logger.logger
#     LOGGER.info('This is a warning message')
