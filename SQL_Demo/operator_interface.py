#!/usr/bin/python3
# -*- coding: utf-8 -*-

from abc import abstractmethod, ABCMeta


class BaseOperator(metaclass=ABCMeta):

    @abstractmethod
    def new(self):
        raise NotImplementedError("new is not implemented.")

    @abstractmethod
    def run(self):
        raise NotImplementedError("new is not implemented.")



