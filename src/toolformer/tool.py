import re
from abc import ABCMeta, abstractmethod


class Tool(metaclass=ABCMeta):
    API_CALL_PREFIX = '['
    API_CALL_SUFFIX = ']'
    RESULT_PREFIX = '->'

    def get_tool_signature(self):
        return '{}{}{}'.format(self.API_CALL_PREFIX, self.get_tool_name().upper(), self.API_CALL_SUFFIX)

    def text_has_api_call(self, text) -> bool:
        # TODO: cleanup
        return len(re.match(r'\[{}\(.*\)\]'.format(self.get_tool_name().upper()), text).groups()) > 0

    @abstractmethod
    def get_tool_name(self):
        pass

    @abstractmethod
    def get_prompt_template(self) -> str:
        pass

    @abstractmethod
    def run(self, input: str) -> str:
        pass