import re

# 전역 변수로 자주 사용되는 정규표현식 패턴 컴파일
SPACE_PATTERN = re.compile(r'\s+')
SLASH_PATTERN = re.compile(r'/+')
RECOMMENDATION_PATTERN = re.compile(r'\S+/\S+|\S+')

def remove_unnecessary_tokens(text, unnecessary_tokens=None):
    """
    불필요한 토큰을 제거하는 함수

    Args:
        text (str): 처리할 텍스트
        unnecessary_tokens (list, optional): 제거할 불필요한 토큰 리스트. 
                                             기본값은 None이며, 이 경우 미리 정의된 리스트를 사용합니다.

    Returns:
        str: 처리된 텍스트
    """
    default_unnecessary_tokens = [
        r'\b어~', r'\b그~', r'\b음~', r'\b아~', r'\b이~', r'\b에~', 
        r'\b저~', r'\b뭐~', r'\b으~', r'\b자~', r'\b응~', r'\b야~',
        r'\b\S*[xX]\S*\b',
    ]

    tokens_to_remove = unnecessary_tokens if unnecessary_tokens is not None else default_unnecessary_tokens
    
    # 모든 토큰을 하나의 정규표현식으로 결합
    combined_pattern = re.compile('|'.join(tokens_to_remove))
    
    # 불필요한 토큰들을 한 번에 제거
    text = combined_pattern.sub('', text)

    # 연속된 공백을 하나의 공백으로
    text = SPACE_PATTERN.sub(' ', text).strip()

    # 연속된 /를 하나의 /로
    text = SLASH_PATTERN.sub('/', text).strip()

    return text

def remove_recommendation(text):
    """
    슬래시(/)로 구분된 단어 제안에서 앞쪽 단어만 선택하는 함수

    Args:
        text (str): 처리할 텍스트

    Returns:
        str: 처리된 텍스트
    """
    return RECOMMENDATION_PATTERN.sub(lambda m: m.group(0).split('/')[0], text)

def remove_special_characters(text, keep_chars=None):
    """
    지정된 특수문자를 제외한 모든 특수문자를 제거하는 함수

    Args:
        text (str): 처리할 한글 대화 텍스트
        keep_chars (set, optional): 유지할 특수문자 집합. 
                                    기본값은 None이며, 이 경우 미리 정의된 집합을 사용합니다.

    Returns:
        str: 처리된 텍스트
    """
    default_keep_chars = set([':', '.', '?', '%', '!', ','])
    chars_to_keep = keep_chars if keep_chars is not None else default_keep_chars

    pattern = re.compile(f'[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\s{"".join(re.escape(c) for c in chars_to_keep)}]')

    cleaned_text = pattern.sub('', text)
    cleaned_text = SPACE_PATTERN.sub(' ', cleaned_text).strip()

    return cleaned_text

# 테스트 코드는 그대로 유지

if __name__ == "__main__":
    print("=== 함수 테스트 시작 ===\n")

    # Test remove_unnecessary_tokens
    print("1. remove_unnecessary_tokens 테스트")
    test_text1 = "어~ 이것은 그~ 테스트 문장입니다. 음~ xTest와 Xylophone은 제거되어야 해요."
    print("원본:", test_text1)
    print("기본 설정 결과:", remove_unnecessary_tokens(test_text1))
    custom_tokens = [r'\b테스트\b', r'\b제거\b']
    print("사용자 정의 토큰 결과:", remove_unnecessary_tokens(test_text1, unnecessary_tokens=custom_tokens))
    print()

    # Test remove_recommendation
    print("2. remove_recommendation 테스트")
    test_text2 = "이것은/저것은 테스트/실험 문장입니다. 선택지1/선택지2 중 하나를 고르세요."
    print("원본:", test_text2)
    print("처리 결과:", remove_recommendation(test_text2))
    print()

    # Test remove_special_characters
    print("3. remove_special_characters 테스트")
    test_text3 = "안녕하세요! 이것은, 테스트(test) 문장입니다. 특수문자 (예: @#$%)를 제거해야 합니다. 그런데 !, ?, ., ,, :, % 는 유지해야 합니다."
    print("원본:", test_text3)
    print("기본 설정 결과:", remove_special_characters(test_text3))
    custom_keep_chars = set([':', '.', '?', '%', '!', '(', ')'])
    print("사용자 정의 유지 문자 결과:", remove_special_characters(test_text3, keep_chars=custom_keep_chars))
    print()

    print("=== 함수 테스트 종료 ===")
