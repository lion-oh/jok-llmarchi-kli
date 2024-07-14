
SYSTEM_PROMPT_1 = '''
You are a helpful AI assistant. 
Please answer the user's questions kindly. 
당신은 유능한 AI 어시스턴트 입니다. 
사용자의 질문에 대해 친절하게 답변해주세요.
'''

USER_PROMPT_1 = lambda subject_keyword, user1, user2: f'''
[Question] \n 사용자들의 대화의 주제는 {subject_keyword}이다. 주제에 대한 대화를 요약하라.
- 첫번째로 {subject_keyword} 주제에 대해서 요약하라.
- 두번째로 {user1}의 대화 내용을 요약하라.
- 세번째로 {user2}의 대화 내용을 요약하라.
'''
