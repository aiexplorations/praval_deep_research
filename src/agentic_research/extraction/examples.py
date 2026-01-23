"""
Few-shot examples for LangExtract-based academic paper extraction.

These examples train the LLM to extract structured entities from
research papers with proper source grounding.
"""

from typing import List, Dict, Any

# Example 1: NLP Paper (BERT-style)
EXAMPLE_NLP_PAPER = {
    "text": """
    We introduce BERT, a new language representation model that stands for
    Bidirectional Encoder Representations from Transformers. Unlike previous
    language representation models, BERT is designed to pre-train deep
    bidirectional representations from unlabeled text by jointly conditioning
    on both left and right context in all layers. As a result, the pre-trained
    BERT model can be fine-tuned with just one additional output layer to
    create state-of-the-art models for a wide range of tasks, such as question
    answering and language inference, without substantial task-specific
    architecture modifications.

    We evaluate on eleven natural language processing tasks and demonstrate
    new state-of-the-art results on all of them, including pushing the GLUE
    score to 80.5% (7.7 point absolute improvement), MultiNLI accuracy to
    86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1
    to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1
    (5.1 point absolute improvement).

    Our work builds on the Transformer architecture introduced in Vaswani et
    al. (2017), which has been shown to be effective for various NLP tasks.
    We extend this by using masked language modeling, inspired by the Cloze
    task, combined with next sentence prediction for pre-training.

    A key limitation of our approach is the computational cost. Pre-training
    BERT-large requires 64 TPU chips for 4 days, which may not be accessible
    to all researchers.
    """,
    "extractions": [
        {
            "type": "method",
            "name": "BERT",
            "content": "BERT is a new language representation model that pre-trains deep bidirectional representations from unlabeled text by conditioning on both left and right context.",
            "source_span": {"start": 17, "end": 456, "text": "BERT, a new language representation model..."},
            "confidence": 0.95,
            "attributes": {
                "domain": "NLP",
                "novelty": "Bidirectional pre-training unlike previous unidirectional models",
                "components": ["Transformer encoder", "Masked language modeling", "Next sentence prediction"],
            },
        },
        {
            "type": "dataset",
            "name": "GLUE",
            "content": "General Language Understanding Evaluation benchmark",
            "source_span": {"start": 680, "end": 750, "text": "pushing the GLUE score to 80.5%..."},
            "confidence": 0.9,
            "attributes": {
                "usage": "Evaluation benchmark",
                "domain": "Natural language understanding",
            },
        },
        {
            "type": "dataset",
            "name": "SQuAD v1.1",
            "content": "Stanford Question Answering Dataset version 1.1",
            "source_span": {"start": 820, "end": 900, "text": "SQuAD v1.1 question answering Test F1 to 93.2..."},
            "confidence": 0.9,
            "attributes": {
                "usage": "Evaluation benchmark for question answering",
                "domain": "Reading comprehension",
            },
        },
        {
            "type": "finding",
            "name": "State-of-the-art on 11 NLP tasks",
            "content": "BERT achieves state-of-the-art results on eleven natural language processing tasks",
            "source_span": {"start": 620, "end": 970, "text": "We evaluate on eleven natural language processing tasks..."},
            "confidence": 0.95,
            "attributes": {
                "evidence": "Experiments on GLUE, MultiNLI, SQuAD benchmarks",
                "significance": "Demonstrates effectiveness of bidirectional pre-training",
            },
        },
        {
            "type": "metric",
            "name": "F1 Score",
            "content": "SQuAD v1.1 Test F1 score of 93.2",
            "source_span": {"start": 820, "end": 920, "text": "SQuAD v1.1 question answering Test F1 to 93.2..."},
            "confidence": 0.95,
            "attributes": {
                "value": "93.2",
                "dataset": "SQuAD v1.1",
                "comparison": "1.5 point absolute improvement over prior work",
            },
        },
        {
            "type": "citation",
            "name": "Transformer",
            "content": "Reference to the original Transformer architecture paper",
            "source_span": {"start": 1050, "end": 1180, "text": "builds on the Transformer architecture introduced in Vaswani et al. (2017)..."},
            "confidence": 0.9,
            "attributes": {
                "cited_work": "Attention Is All You Need",
                "relationship": "builds_upon",
                "context": "BERT extends the Transformer architecture for bidirectional pre-training",
            },
        },
        {
            "type": "limitation",
            "name": "Computational cost",
            "content": "Pre-training BERT-large requires 64 TPU chips for 4 days",
            "source_span": {"start": 1320, "end": 1480, "text": "A key limitation of our approach is the computational cost..."},
            "confidence": 0.9,
            "attributes": {
                "impact": "May not be accessible to all researchers",
                "mitigation": None,
            },
        },
    ],
}

# Example 2: Computer Vision Paper (ResNet-style)
EXAMPLE_CV_PAPER = {
    "text": """
    We present ResNet, a residual learning framework for training very deep
    neural networks. Our approach introduces skip connections that allow
    gradients to flow directly through the network, addressing the degradation
    problem where deeper networks have higher training error.

    We evaluate on the ImageNet dataset, which contains 1.28 million training
    images across 1000 classes. Our 152-layer ResNet achieves a top-5 error
    rate of 3.57% on the ImageNet test set, winning the ILSVRC 2015
    classification task.

    We also demonstrate strong generalization to object detection on the COCO
    dataset, achieving 59.1% mAP on the test-dev set, outperforming all
    previous approaches.

    The key insight is that identity mappings through skip connections make
    optimization of very deep networks feasible. However, our approach requires
    careful initialization and batch normalization for stable training.
    """,
    "extractions": [
        {
            "type": "method",
            "name": "ResNet",
            "content": "ResNet is a residual learning framework using skip connections to enable training of very deep neural networks",
            "source_span": {"start": 12, "end": 220, "text": "ResNet, a residual learning framework..."},
            "confidence": 0.95,
            "attributes": {
                "domain": "Computer Vision",
                "novelty": "Skip connections for gradient flow in very deep networks",
                "components": ["Skip connections", "Residual blocks", "Batch normalization"],
            },
        },
        {
            "type": "dataset",
            "name": "ImageNet",
            "content": "ImageNet dataset with 1.28 million training images across 1000 classes",
            "source_span": {"start": 280, "end": 400, "text": "ImageNet dataset, which contains 1.28 million training images..."},
            "confidence": 0.95,
            "attributes": {
                "size": "1.28 million images",
                "domain": "Image classification",
                "usage": "Training and evaluation",
            },
        },
        {
            "type": "metric",
            "name": "Top-5 Error Rate",
            "content": "152-layer ResNet achieves 3.57% top-5 error rate on ImageNet",
            "source_span": {"start": 400, "end": 520, "text": "152-layer ResNet achieves a top-5 error rate of 3.57%..."},
            "confidence": 0.95,
            "attributes": {
                "value": "3.57%",
                "dataset": "ImageNet test set",
                "comparison": "Won ILSVRC 2015 classification",
            },
        },
        {
            "type": "finding",
            "name": "Skip connections enable deep network training",
            "content": "Identity mappings through skip connections make optimization of very deep networks feasible",
            "source_span": {"start": 620, "end": 780, "text": "The key insight is that identity mappings..."},
            "confidence": 0.9,
            "attributes": {
                "significance": "Solves the degradation problem in deep networks",
            },
        },
    ],
}

# Example 3: ML Theory Paper (Attention-style)
EXAMPLE_THEORY_PAPER = {
    "text": """
    We propose the Transformer, a novel architecture based entirely on
    attention mechanisms, dispensing with recurrence and convolutions
    entirely. The core component is the scaled dot-product attention,
    computed as Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V.

    We introduce multi-head attention, which allows the model to jointly
    attend to information from different representation subspaces at
    different positions: MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O.

    Experiments on the WMT 2014 English-to-German translation task show
    that our model achieves 28.4 BLEU, outperforming previous best results
    including ensembles. On English-to-French, we achieve 41.8 BLEU,
    establishing a new single-model state-of-the-art.

    The Transformer is fundamentally different from RNN-based approaches
    like the sequence-to-sequence model with attention (Bahdanau et al.,
    2015). Unlike RNNs, the Transformer allows for significantly more
    parallelization during training.
    """,
    "extractions": [
        {
            "type": "method",
            "name": "Transformer",
            "content": "Novel architecture based entirely on attention mechanisms without recurrence or convolutions",
            "source_span": {"start": 12, "end": 150, "text": "the Transformer, a novel architecture based entirely on attention mechanisms..."},
            "confidence": 0.95,
            "attributes": {
                "domain": "Sequence modeling",
                "novelty": "First architecture using only attention, no recurrence or convolution",
                "components": ["Scaled dot-product attention", "Multi-head attention", "Position encoding"],
            },
        },
        {
            "type": "method",
            "name": "Scaled dot-product attention",
            "content": "Attention mechanism computed as softmax(QK^T/sqrt(d_k))V",
            "source_span": {"start": 180, "end": 320, "text": "scaled dot-product attention, computed as Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V..."},
            "confidence": 0.9,
            "attributes": {
                "domain": "Attention mechanisms",
                "description": "Core attention computation with scaling factor",
            },
        },
        {
            "type": "metric",
            "name": "BLEU Score",
            "content": "28.4 BLEU on WMT 2014 English-to-German translation",
            "source_span": {"start": 520, "end": 650, "text": "our model achieves 28.4 BLEU, outperforming previous best results..."},
            "confidence": 0.95,
            "attributes": {
                "value": "28.4",
                "dataset": "WMT 2014 English-to-German",
                "comparison": "Outperforms previous best including ensembles",
            },
        },
        {
            "type": "citation",
            "name": "Sequence-to-sequence with attention",
            "content": "Reference to prior sequence-to-sequence model with attention",
            "source_span": {"start": 750, "end": 900, "text": "sequence-to-sequence model with attention (Bahdanau et al., 2015)..."},
            "confidence": 0.85,
            "attributes": {
                "cited_work": "Neural Machine Translation by Jointly Learning to Align and Translate",
                "relationship": "contrasts",
                "context": "Transformer differs from RNN-based approaches",
            },
        },
    ],
}


def get_extraction_examples() -> List[Dict[str, Any]]:
    """
    Get all extraction examples for few-shot learning.

    Returns:
        List of example dictionaries with text and extractions
    """
    return [
        EXAMPLE_NLP_PAPER,
        EXAMPLE_CV_PAPER,
        EXAMPLE_THEORY_PAPER,
    ]


def get_examples_for_type(extraction_type: str) -> List[Dict[str, Any]]:
    """
    Get examples filtered by extraction type.

    Args:
        extraction_type: Type of extraction (method, dataset, finding, etc.)

    Returns:
        List of relevant extraction examples
    """
    all_examples = get_extraction_examples()
    filtered = []

    for example in all_examples:
        type_extractions = [
            e for e in example["extractions"]
            if e["type"] == extraction_type
        ]
        if type_extractions:
            filtered.append({
                "text": example["text"],
                "extractions": type_extractions,
            })

    return filtered


def format_examples_for_prompt(examples: List[Dict[str, Any]], max_examples: int = 2) -> str:
    """
    Format examples into a prompt-friendly string.

    Args:
        examples: List of example dictionaries
        max_examples: Maximum number of examples to include

    Returns:
        Formatted string for inclusion in LLM prompt
    """
    formatted_parts = []

    for i, example in enumerate(examples[:max_examples]):
        text_preview = example["text"][:500] + "..." if len(example["text"]) > 500 else example["text"]

        extractions_text = []
        for ext in example["extractions"][:3]:  # Limit extractions per example
            ext_str = f"""
  - Type: {ext['type']}
    Name: {ext['name']}
    Content: {ext['content'][:200]}...
    Source: "{ext['source_span']['text'][:100]}..."
"""
            extractions_text.append(ext_str)

        formatted_parts.append(f"""
Example {i + 1}:
Text: {text_preview}

Extractions:
{''.join(extractions_text)}
""")

    return "\n".join(formatted_parts)


# Prompt template for paper extraction
EXTRACTION_PROMPT_TEMPLATE = """
Extract structured entities from the following academic research paper text.
For each entity, provide:
1. Type (method, dataset, finding, citation, metric, limitation)
2. Name (short identifier)
3. Content (detailed description)
4. Source span (exact text location for grounding)
5. Relevant attributes based on entity type

Focus on:
- Novel methods or techniques proposed
- Datasets used for evaluation
- Key findings and results with evidence
- Important citations and their relationship
- Quantitative metrics and comparisons
- Acknowledged limitations

{examples}

Now extract entities from this paper:

Text:
{text}

Return extractions in JSON format with the structure:
{{
  "extractions": [
    {{
      "type": "method|dataset|finding|citation|metric|limitation",
      "name": "short name",
      "content": "detailed description",
      "source_span": {{
        "start": char_offset,
        "end": char_offset,
        "text": "exact source text"
      }},
      "confidence": 0.0-1.0,
      "attributes": {{}}
    }}
  ]
}}
"""
