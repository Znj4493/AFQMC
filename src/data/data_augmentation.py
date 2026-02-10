"""
数据增强模块 - EDA (Easy Data Augmentation)

实现4种文本增强操作：
1. Synonym Replacement (SR) - 同义词替换
2. Random Insertion (RI) - 随机插入
3. Random Swap (RS) - 随机交换
4. Random Deletion (RD) - 随机删除

参考论文: "EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks"
"""

import random
import jieba
import numpy as np
from typing import List, Tuple


# ============================================
# 停用词列表（简化版）
# ============================================
STOP_WORDS = set([
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很',
    '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '么', '吗', '呢',
    '啊', '哦', '嗯', '吧', '呀', '啦', '嘛', '呗', '。', '，', '！', '？', '、', '；', '：', '"', '"',
    ''', ''', '（', '）', '【', '】', '《', '》', '…', '—'
])


# ============================================
# 同义词词典（简化版）
# ============================================
# 在实际应用中，可以使用更完整的同义词库，如：
# - HIT-SCIR同义词词林（https://github.com/guotong1988/chinese_dictionary）
# - 或使用word2vec/BERT获取语义相似词
SYNONYM_DICT = {
    '手机': ['电话', '移动电话', '手提电话'],
    '电脑': ['计算机', '电子计算机'],
    '好': ['不错', '很好', '优秀', '棒'],
    '快': ['迅速', '快速', '飞快'],
    '慢': ['缓慢', '慢速'],
    '大': ['巨大', '庞大', '宽大'],
    '小': ['微小', '细小', '较小'],
    '高': ['高大', '高耸'],
    '低': ['低矮', '矮小'],
    '新': ['崭新', '全新'],
    '旧': ['陈旧', '老旧'],
    '多': ['很多', '许多', '大量'],
    '少': ['较少', '稀少'],
    '便宜': ['实惠', '廉价'],
    '贵': ['昂贵', '高价'],
    '漂亮': ['美丽', '好看', '美观'],
    '难看': ['丑陋', '不好看'],
    '喜欢': ['喜爱', '热爱'],
    '讨厌': ['厌恶', '不喜欢'],
    '购买': ['买', '采购', '选购'],
    '支付': ['付款', '缴费', '交费'],
    '贷款': ['借款', '借贷'],
    '还款': ['还贷', '偿还'],
    '提现': ['取现', '取款'],
    '充值': ['充钱', '存款'],
    '账户': ['账号', '帐户'],
    '金额': ['数额', '钱数'],
    '利息': ['利率', '息金'],
    '手续费': ['费用', '服务费'],
}


def get_synonyms(word: str) -> List[str]:
    """
    获取词的同义词列表

    参数:
        word: 输入词

    返回:
        同义词列表（不包含原词）
    """
    return SYNONYM_DICT.get(word, [])


def get_words(sentence: str) -> List[str]:
    """
    中文分词，过滤停用词

    参数:
        sentence: 输入句子

    返回:
        分词列表（不含停用词）
    """
    words = jieba.lcut(sentence)
    # 过滤停用词和空白字符
    words = [w for w in words if w.strip() and w not in STOP_WORDS]
    return words


# ============================================
# EDA 核心函数
# ============================================

def synonym_replacement(sentence: str, n: int) -> str:
    """
    同义词替换 (Synonym Replacement)

    随机选择n个非停用词，将其替换为同义词

    参数:
        sentence: 输入句子
        n: 替换的词数

    返回:
        增强后的句子
    """
    words = jieba.lcut(sentence)
    # 找到可替换的词（非停用词且有同义词）
    replaceable_indices = []
    for i, word in enumerate(words):
        if word not in STOP_WORDS and get_synonyms(word):
            replaceable_indices.append(i)

    # 如果没有可替换的词，返回原句
    if not replaceable_indices:
        return sentence

    # 随机选择n个位置进行替换
    n = min(n, len(replaceable_indices))
    replace_indices = random.sample(replaceable_indices, n)

    # 执行替换
    new_words = words.copy()
    for idx in replace_indices:
        word = words[idx]
        synonyms = get_synonyms(word)
        if synonyms:
            new_words[idx] = random.choice(synonyms)

    return ''.join(new_words)


def random_insertion(sentence: str, n: int) -> str:
    """
    随机插入 (Random Insertion)

    随机选择n个非停用词，在句子的随机位置插入其同义词

    参数:
        sentence: 输入句子
        n: 插入次数

    返回:
        增强后的句子
    """
    words = jieba.lcut(sentence)

    for _ in range(n):
        # 找到有同义词的词
        candidate_words = [w for w in words if w not in STOP_WORDS and get_synonyms(w)]

        if not candidate_words:
            continue

        # 随机选一个词
        word = random.choice(candidate_words)
        synonyms = get_synonyms(word)

        if synonyms:
            # 随机选一个同义词
            synonym = random.choice(synonyms)
            # 在随机位置插入
            random_idx = random.randint(0, len(words))
            words.insert(random_idx, synonym)

    return ''.join(words)


def random_swap(sentence: str, n: int) -> str:
    """
    随机交换 (Random Swap)

    随机交换句子中两个词的位置，重复n次

    参数:
        sentence: 输入句子
        n: 交换次数

    返回:
        增强后的句子
    """
    words = jieba.lcut(sentence)

    # 句子太短无法交换
    if len(words) < 2:
        return sentence

    for _ in range(n):
        # 随机选择两个不同的位置
        idx1, idx2 = random.sample(range(len(words)), 2)
        # 交换
        words[idx1], words[idx2] = words[idx2], words[idx1]

    return ''.join(words)


def random_deletion(sentence: str, p: float) -> str:
    """
    随机删除 (Random Deletion)

    以概率p随机删除句子中的每个词

    参数:
        sentence: 输入句子
        p: 删除概率（0-1之间）

    返回:
        增强后的句子
    """
    words = jieba.lcut(sentence)

    # 如果句子只有一个词，不删除
    if len(words) == 1:
        return sentence

    # 随机删除
    new_words = []
    for word in words:
        # 以概率1-p保留该词
        if random.random() > p:
            new_words.append(word)

    # 如果全部删除了，至少保留一个词
    if len(new_words) == 0:
        return random.choice(words)

    return ''.join(new_words)


# ============================================
# 组合增强函数
# ============================================

def eda(sentence: str, alpha: float = 0.1, num_aug: int = 4) -> List[str]:
    """
    EDA综合增强

    对一个句子应用4种增强操作，生成多个增强样本

    参数:
        sentence: 输入句子
        alpha: 增强强度参数（0-1），控制每种操作影响的词数比例
        num_aug: 生成的增强样本数量

    返回:
        增强后的句子列表（不包含原句）
    """
    # 计算操作次数
    words = get_words(sentence)
    num_words = len(words)

    if num_words == 0:
        return []

    # 每种操作影响的词数（至少1个）
    n = max(1, int(alpha * num_words))

    # 生成增强样本
    augmented_sentences = []

    # 每种操作生成 num_aug/4 个样本（如果num_aug不是4的倍数，SR多生成）
    num_per_technique = num_aug // 4
    remaining = num_aug % 4

    # 同义词替换
    for _ in range(num_per_technique + (1 if remaining > 0 else 0)):
        aug_sent = synonym_replacement(sentence, n)
        if aug_sent != sentence:  # 确保生成的句子与原句不同
            augmented_sentences.append(aug_sent)

    # 随机插入
    for _ in range(num_per_technique + (1 if remaining > 1 else 0)):
        aug_sent = random_insertion(sentence, n)
        if aug_sent != sentence:
            augmented_sentences.append(aug_sent)

    # 随机交换
    for _ in range(num_per_technique + (1 if remaining > 2 else 0)):
        aug_sent = random_swap(sentence, n)
        if aug_sent != sentence:
            augmented_sentences.append(aug_sent)

    # 随机删除
    for _ in range(num_per_technique):
        aug_sent = random_deletion(sentence, alpha)
        if aug_sent != sentence:
            augmented_sentences.append(aug_sent)

    # 去重
    augmented_sentences = list(set(augmented_sentences))

    # 如果生成的样本不够，用SR补充
    while len(augmented_sentences) < num_aug:
        aug_sent = synonym_replacement(sentence, n)
        if aug_sent not in augmented_sentences and aug_sent != sentence:
            augmented_sentences.append(aug_sent)
        else:
            # 如果SR无法生成新样本，尝试其他方法
            aug_sent = random_insertion(sentence, n)
            if aug_sent not in augmented_sentences and aug_sent != sentence:
                augmented_sentences.append(aug_sent)
            else:
                break  # 无法生成更多样本

    return augmented_sentences[:num_aug]


# ============================================
# 针对句子对的增强函数
# ============================================

def augment_sentence_pair(
    sentence1: str,
    sentence2: str,
    label: int,
    alpha: float = 0.1,
    num_aug: int = 1,
    augment_both: bool = False
) -> List[Tuple[str, str, int]]:
    """
    对句子对进行数据增强

    策略：
    - 只增强sentence1，或
    - 只增强sentence2，或
    - 同时增强两个句子

    参数:
        sentence1: 第一个句子
        sentence2: 第二个句子
        label: 标签（0或1）
        alpha: 增强强度
        num_aug: 每个句子生成的增强样本数
        augment_both: 是否同时增强两个句子

    返回:
        增强后的句子对列表: [(sent1, sent2, label), ...]
    """
    augmented_pairs = []

    if augment_both:
        # 同时增强两个句子
        aug_sent1_list = eda(sentence1, alpha=alpha, num_aug=num_aug)
        aug_sent2_list = eda(sentence2, alpha=alpha, num_aug=num_aug)

        # 配对（取较短列表的长度）
        min_len = min(len(aug_sent1_list), len(aug_sent2_list))
        for i in range(min_len):
            augmented_pairs.append((aug_sent1_list[i], aug_sent2_list[i], label))
    else:
        # 只增强sentence1
        aug_sent1_list = eda(sentence1, alpha=alpha, num_aug=num_aug)
        for aug_sent1 in aug_sent1_list:
            augmented_pairs.append((aug_sent1, sentence2, label))

        # 只增强sentence2
        aug_sent2_list = eda(sentence2, alpha=alpha, num_aug=num_aug)
        for aug_sent2 in aug_sent2_list:
            augmented_pairs.append((sentence1, aug_sent2, label))

    return augmented_pairs


# ============================================
# 测试代码
# ============================================

if __name__ == '__main__':
    # 测试单个句子增强
    test_sentence = "这个手机质量很好，价格便宜"

    print("=" * 60)
    print(f"原句: {test_sentence}")
    print("=" * 60)

    print("\n同义词替换 (SR):")
    for i in range(3):
        print(f"  {i+1}. {synonym_replacement(test_sentence, n=2)}")

    print("\n随机插入 (RI):")
    for i in range(3):
        print(f"  {i+1}. {random_insertion(test_sentence, n=2)}")

    print("\n随机交换 (RS):")
    for i in range(3):
        print(f"  {i+1}. {random_swap(test_sentence, n=2)}")

    print("\n随机删除 (RD):")
    for i in range(3):
        print(f"  {i+1}. {random_deletion(test_sentence, p=0.2)}")

    print("\nEDA综合增强:")
    aug_sentences = eda(test_sentence, alpha=0.1, num_aug=4)
    for i, sent in enumerate(aug_sentences):
        print(f"  {i+1}. {sent}")

    print("\n" + "=" * 60)
    print("句子对增强测试")
    print("=" * 60)

    sent1 = "花呗如何还款"
    sent2 = "花呗怎么还钱"
    label = 1

    print(f"原句对:")
    print(f"  Sentence1: {sent1}")
    print(f"  Sentence2: {sent2}")
    print(f"  Label: {label}")

    print(f"\n增强后的句子对 (只增强单侧):")
    aug_pairs = augment_sentence_pair(sent1, sent2, label, alpha=0.1, num_aug=2, augment_both=False)
    for i, (s1, s2, l) in enumerate(aug_pairs):
        print(f"  {i+1}. [{s1}] | [{s2}] | {l}")

    print(f"\n增强后的句子对 (同时增强):")
    aug_pairs = augment_sentence_pair(sent1, sent2, label, alpha=0.1, num_aug=2, augment_both=True)
    for i, (s1, s2, l) in enumerate(aug_pairs):
        print(f"  {i+1}. [{s1}] | [{s2}] | {l}")
