import re
from abc import ABCMeta, abstractmethod


class Tool(metaclass=ABCMeta):
    API_CALL_PREFIX = '['
    API_CALL_SUFFIX = ']'
    RESULT_PREFIX = '->'

    def get_tool_signature(self):
        return '{}{}{}'.format(self.API_CALL_PREFIX, self.get_tool_name().upper(), self.API_CALL_SUFFIX)

    def get_tool_regex(self, match_before=False):
        result = r'\[{}\(.*\)\]'.format(self.get_tool_name().upper())
        if match_before:
            result = r'^.*' + result
        return result

    def text_has_api_call(self, text) -> bool:
        return re.match(self.get_tool_regex(), text) is not None

    def get_api_call_from_text(self, text) -> str:
        result = re.search('^.*(?P<api_call>{})'.format(self.get_tool_regex()), text)
        return result.groupdict()['api_call']

    def get_text_before_api_call(self, text) -> str:
        # TODO: refactor
        result = re.search('^.*(?P<api_call>{})'.format(self.get_tool_regex()), text)
        return text[:result.span('api_call')[0]]

    def get_text_after_api_call(self, text) -> str:
        result = re.search('^.*(?P<api_call>{})'.format(self.get_tool_regex()), text)
        return text[result.span('api_call')[1]:]

    @ abstractmethod
    def get_tool_name(self):
        pass

    @abstractmethod
    def get_prompt_template(self) -> str:
        pass

    @abstractmethod
    def run(self, input: str) -> str:
        pass
