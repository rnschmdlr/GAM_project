# Chromatin Architecture Entropy

This repository contains the code implemented for the thesis titled "Transfer Entropy Reveals Directed Spatial Interactions in Time-resolved Processes of Chromatin Organisation" The goal of this research is to provide efficient methods for quantifying chromatin organization changes without making assumptions about the underlying processes.

## Abstract

Understanding the dynamic nature of chromatin structure is crucial for unraveling the mechanisms behind gene regulation and cell function. This thesis introduces a novel approach using transfer entropy, a concept from information theory, to analyze chromatin dynamics captured in Genome Architecture Mapping (GAM) data. Unlike static measures, our approach reveals directional influences between genomic loci.

We present three parametric models based on the loop extrusion framework, demonstrating the effectiveness of transfer entropy in deciphering the complex dynamics of simulated chromatin formation. Our findings indicate that this method can identify key genomic loci and their directed influences in our models.

Moreover, we explore how transfer entropy could enhance pseudotime inference techniques, highlighting its potential as an innovative orthogonal method to integrate chromatin organization data. Our results show that transfer entropy can differentiate the direction and progression of chromatin formation processes, refining the accuracy and interpretability of pseudotime ordering.

## Repository Structure

- `data/`: Includes the three parametric sample models used in this thesis.
- `src/`: Contains the source code for implementing shannon entropy as well as transfer entropy analysis on GAM data. 
`/toymodel`contains the source code for generating genome architecture models based on the loop extrusion framework.
`/project_work` contains depreciated, self-contained code of an earlier related project.

## Usage

Feel free to adapt the provided code for your own chromatin organization data and configuration or geenrate your own models. For analysis, start with `src/entropy/transfer_entropy/pipeline.py`. Update the path to your input data and modify the parameters as needed. All interactive files provide additional analysis workflows.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
