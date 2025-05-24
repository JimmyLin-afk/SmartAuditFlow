# SmartAuditFlow - Smart Contract Audit Platform 

<p>
<img align="right" width="180"  src="./media/front_view.png"> 
</p>

**SmartAuditFlow** is a dynamic and adaptive framework for automated smart contract auditing, leveraging Large Language Models (LLMs) and workflow-driven strategies to deliver reliable, precise, and scalable security analysis.

## Overview

**SmartAuditFlow** addresses these limitations by orchestrating LLMs within a structured, multi-stage workflow, enabling dynamic audit plan generation, iterative reasoning, and integration of external tools for enhanced vulnerability detection.

Key innovations include:

* **Dynamic audit plan customization** for each contract.
* **Iterative execution and refinement** of audit strategies based on intermediate findings.
* **Structured reasoning** and **prompt optimization** to improve LLM performance.
* **Integration with static analyzers** and **Retrieval-Augmented Generation (RAG)** for contextual enrichment.

Experimental results show that SmartAuditFlow achieves **100% accuracy** on common vulnerability benchmarks and identifies **13 additional CVEs** missed by other methods, demonstrating superior adaptability and utility.

===========================================
## Getting Started

The source code and instructions are available at:
[https://github.com/JimmyLin-afk/SmartAuditFlow](https://github.com/JimmyLin-afk/SmartAuditFlow)

1. **Clone the repository**

   ```bash
   git clone https://github.com/JimmyLin-afk/SmartAuditFlow.git
   cd SmartAuditFlow
   ```

2. **Install dependencies**
   (See [requirements.txt](./requirements.txt) or the project documentation.)

3. **Run the demo**
   Follow the instructions in the [documentation](./docs/) to perform your first smart contract audit.

## Citation

If you use SmartAuditFlow in your research, please cite:

```
@misc{wei2025adaptiveplanexecuteframeworksmart,
      title={Adaptive Plan-Execute Framework for Smart Contract Security Auditing}, 
      author={Zhiyuan Wei and Jing Sun and Zijian Zhang and Zhe Hou and Zixiao Zhao},
      year={2025},
      eprint={2505.15242},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2505.15242}, 
}
```
