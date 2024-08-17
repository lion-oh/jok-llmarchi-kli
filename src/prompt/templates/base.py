_PROMPT_PREFIX = '''You are a helpful AI assistant. Please answer the user's questions kindly. '''

_PROMPT_SUFFIX = "[Question]\n위 {keyword} 주제에 대한 대화(Conversation)를 요약(Summarize)해주세요. Let's think step by step"

DEFAULT_PROMPT = """Conversation format과 Summarize format을 참고해서 답변을 생성해주고 Condition을 잘 따라줘.
<Conversation>
화자SD0000001: 대화내용
화자SD0000002: 대화내용
화자SD0000001: 대화내용
화자SD0000002: 대화내용
...
</Conversation>

<Summarize>
이 대화에서 화자들은 주제에 대해 이야기했습니다. SD0000001는 [주요 내용]이라고 말했습니다. 그리고 [구체적인 예시 및 이유]를 언급했습니다. SD0000002는 [주요 내용]이라고 말했고, [구체적인 예시 및 이유]를 설명했습니다
</Summarize>

<Condition>.
- 요약할 때는 가장 먼저 어떤 주제에 대해서 이야기 했는지 언급.
- Conversation에서 먼저 등장하는 화자부터 대화를 요약.
- 화자는 'SD0000001'의 형태를 띄고 있음.
</Condition>
"""