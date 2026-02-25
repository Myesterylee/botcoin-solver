# BOTCOIN Solver

确定性 BOTCOIN 挖矿挑战求解器，零 LLM 依赖，纯 Python 实现。

## 功能

- 多遍文档解析：支持 ENTITY/FILING/Transcript 等多种文档格式
- 正则驱动的问题分类与确定性计算
- 自动约束求解：字数、Acrostic、质数、方程式、禁用字母等
- 内置验证引擎

## 准确率

在 1723 道题库中随机抽样 200 道测试：
- 完全通过率：93.5%（187/200）
- 约束通过率：98.3%（1770/1800）

## 使用方法

### 命令行

```bash
echo '{"doc":"...","questions":[...],"constraints":[...],"companies":[...]}' | python botcoin_solver.py
```

### Python API

```python
from botcoin_solver import solve

challenge = {
    "doc": "...",
    "questions": ["Which company had the highest revenue?", ...],
    "constraints": ["Write EXACTLY 16 words", ...],
    "companies": ["Acme Corp", "Beta Inc", ...]
}

artifact = solve(challenge)
print(artifact)
```

### 批量测试

```bash
# 需要先下载题库 challenges.jsonl
python batch_test.py 100 42   # 测试100道，seed=42
```

## 约束类型

| 约束 | 说明 |
|------|------|
| C0 | 精确字数 |
| C1-C3 | 包含特定元素（城市、CEO姓氏、国家） |
| C4 | 质数：nextPrime((员工数 mod 100) + 11) |
| C5 | 方程式：A+B=C 基于季度收入 |
| C6 | Acrostic：前N词首字母拼写 |
| C7 | 禁用字母 |

## 测试

```bash
python -m pytest test_bug_exploration.py test_preservation.py -v
```

## License

MIT
