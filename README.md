# ViOCE

The code implements the approach presented in the paper `Towards Knowledge-aware Few-shot Learning with Ontology-based n-ball Concept Embeddings`.

# Citation

```
@article{jayathilaka2021,
  title={Towards Knowledge-aware Few-shot Learning with Ontology-based n-ball Concept Embeddings},
  author={Jayathilaka, Mirantha and Mu, Tingting and Sattler, Uli},
  booktitle = {20th IEEE International Conference on Machine Learning and Apllications},
  year = {2021}
}
```

# Instructions

### Generating embeddings
- The `el_embeddings` directory contains the code for the generation of n-ball embeddings given an OWL ontology as input. 
- The input ontology should in the OWL Functional Syntaxt format. 
- Run the `generate_embeddings.py` file with the relavant path to the input ontology.

### Few-shot learning
- The `fewshot_model` directory contains the code for traning and validating the vision model informed by the concept embeddings prodcued in the previous step.
- First run `python base_learning.py --d <image data path> --ef <embeddings file path> --cf <class names file path>` for base learning the vision model
- Next run `python few-shot_learning.py --d <image data path> --ef <embeddings file path> --cf <class names file path> --model <trained model path>` for few-shot learning and validation