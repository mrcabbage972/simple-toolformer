from datetime import datetime

from toolformer.tool import Tool


class DateTool(Tool):
    def get_tool_name(self):
        return "DATE"

    def get_prompt_template(self) -> str:
        return """
            Your task is to add calls to a Date API to a piece of text.
            The calls should help you get information required to complete the text.
            Here is an example of an API call:
            
            Input: Joe Biden was born 80 years ago
            Output: Joe Biden was born [DATE()] 80 years ago
            
            Input: {}
            Output: 
            """

    def run(self, input: str) -> str:
        return datetime.today().strftime('%Y-%m-%d')


if __name__ == '__main__':
    print(DateTool().text_has_api_call('aaa [DATE()] bbb'))
    print(DateTool().get_text_before_api_call('aaa [DATE()] bbb'))
    print(DateTool().get_api_call_from_text('aaa [DATE()] bbb'))
    print(DateTool().get_text_after_api_call('aaa [DATE()] bbb'))