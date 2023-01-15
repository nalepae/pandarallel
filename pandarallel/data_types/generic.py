from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, Iterator


class DataType(ABC):
    @staticmethod
    @abstractmethod
    def get_chunks(nb_workers: int, data: Any, **kwargs) -> Iterator[Any]:
        ...

    @staticmethod
    def get_work_extra(data: Any) -> Dict[str, Any]:
        return dict()

    @staticmethod
    @abstractmethod
    def work(
        data: Any,
        user_defined_function: Callable,
        user_defined_function_args: tuple,
        user_defined_function_kwargs: Dict[str, Any],
        extra: Dict[str, Any],
    ) -> Any:
        ...

    @staticmethod
    def get_reduce_extra(
        data: Any, user_defined_function_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        return dict()

    @staticmethod
    @abstractmethod
    def reduce(datas: Iterable[Any], extra: Dict[str, Any]) -> Any:
        ...
