GEORGIAN_TO_ROMAN = {
    'ა': 'a',
    'ბ': 'b',
    'გ': 'g',
    'დ': 'd',
    'ე': 'e',
    'ვ': 'v',
    'ზ': 'z',
    'თ': 't',
    'ი': 'i',
    'კ': 'k',
    'ლ': 'l',
    'მ': 'm',
    'ნ': 'n',
    'ო': 'o',
    'პ': 'p',
    'ჟ': 'zh',
    'რ': 'r',
    'ს': 's',
    'ტ': 't',
    'უ': 'u',
    'ფ': 'p',
    'ქ': 'k',
    'ღ': 'gh',
    'ყ': 'q',
    'შ': 'sh',
    'ჩ': 'ch',
    'ც': 'ts',
    'ძ': 'dz',
    'წ': 'ts',
    'ჭ': 'ch',
    'ხ': 'kh',
    'ჯ': 'j',
    'ჰ': 'h',
}

ROMAN_TO_GEORGIAN = {v: k for k, v in GEORGIAN_TO_ROMAN.items()}


def georgian_to_roman(text: str) -> str:
    result = []
    for char in text:
        if char in GEORGIAN_TO_ROMAN:
            result.append(GEORGIAN_TO_ROMAN[char])
        else:
            result.append(char)
    return ''.join(result)


def roman_to_georgian(text: str) -> str:
    result = []
    i = 0
    while i < len(text):
        found = False
        for length in [2, 1]:
            if i + length <= len(text):
                substr = text[i:i+length]
                if substr in ROMAN_TO_GEORGIAN:
                    result.append(ROMAN_TO_GEORGIAN[substr])
                    i += length
                    found = True
                    break
        if not found:
            result.append(text[i])
            i += 1
    return ''.join(result)


if __name__ == "__main__":
    test_text = "გამარჯობა, მე ვარ ხელოვნური ინტელექტი."
    romanized = georgian_to_roman(test_text)
    print(f"Original: {test_text}")
    print(f"Romanized: {romanized}")
    
    back = roman_to_georgian(romanized)
    print(f"Back to Georgian: {back}")
    print(f"Match: {test_text == back}")

