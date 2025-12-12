# -*- coding: utf-8 -*-

import logging

class Logger():
    def __init__(self, logname, loglevel=logging.DEBUG, loggername=None):
        '''
           指定保存日志的文件路径，日志级别，以及调用文件
           将日志存入到指定的文件中
        '''
        # 创建一个logger
        self.logger = logging.getLogger(loggername)
        self.logger.setLevel(loglevel)

        if not self.logger.handlers:
            # 创建一个handler，用于写入日志文件
            fh = logging.FileHandler(logname, encoding='utf-8')  # 指定编码防止中文乱码
            fh.setLevel(loglevel)

            # 再创建一个handler，用于输出到控制台
            ch = logging.StreamHandler()
            ch.setLevel(loglevel)

            # 定义handler的格式
            formatter = logging.Formatter('%(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)

            # 给logger添加handler
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def getlog(self):
        return self.logger