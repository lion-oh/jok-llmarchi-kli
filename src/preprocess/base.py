import re


def remove_unnecessary_tokens(text):
    # 불필요한 토큰 리스트
    unnecessary_tokens = [
        r'어~', r'아~', r'어\~', r'아\~', r'그\~', r'어\~',
    ]

    # 불필요한 토큰들을 정규표현식으로 제거
    for token in unnecessary_tokens:
        text =re.sub(token, '', text)

    # 연속된 공백을 하나의 공백으로
    text = re.sub(r'\s+', ' ', text).strip()

    return text