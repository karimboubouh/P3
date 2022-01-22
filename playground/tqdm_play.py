from tqdm import tqdm

if __name__ == '__main__':
    pbar = tqdm(["a", "b", "c", "d"], position=0, desc="DESC")
    num_vowels = 0
    for ichar in pbar:
        if ichar in ['a', 'e', 'i', 'o', 'u']:
            num_vowels += 1
        pbar.set_postfix_str("hello")
