#!/bin/env/python3
# coding=utf-8
from __future__ import absolute_import
import unittest
from lmflow.utils.data_utils import load_data, batchlize, answer_extraction
from lmflow.args import DatasetArguments

groundtruth_inputs = ['The Transformer architecture [START_REF]',
          'The Schwarzschild radius is defined as: \\[',
          'A force of 0.6N is applied to an object, which accelerates at 3m/s. What is its mass? <work>', 
          '[START_I_SMILES]',
          '[START_AMINO]GHMQSITAGQKVISKHKNGRFYQCEVVRLTTETFYEVNFDDGSFSDNLYPEDIVSQDCLQFGPPAEGEVVQVRWTDGQVYGAKFVASHPIQMYQVEFEDGSQLVVKRDDVYTLDEELP[END_AMINO] ## Keywords', 
          'The reason why Transformers replaced RNNs was because', 
          'Question: What is the notch signaling pathway?\n\nAnswer:', 
          '# Multi-Head Attention\n\n',
          'Title: Self-Supervised Learning, A Survey\n\nAuthors: John Smith\n\n', 
          'Lecture 1: The Ising Model\n\n', 
          'Information overload is a major obstacle to scientific progress. The explosive growth in scientific literature and data has made it ever harder to discover useful insights in a large mass of information. Today scientific knowledge is accessed through search engines, but they are unable to organize scientific knowledge alone. In this paper we introduce Galactica: a large language model that can store, combine and reason about scientific knowledge. We train on a large scientific corpus of papers, reference material, knowledge bases and many other sources. We outperform existing models on a range of scientific tasks. On technical knowledge probes such as LaTeX equations, Galactica outperforms the latest GPT-3 by 68.2% versus 49.0%. Galactica also performs well on reasoning, outperforming Chinchilla on mathematical MMLU by 41.3% to 35.7%, and PaLM 540B on MATH with a score of 20.4% versus 8.8%. It also sets a new state-of-the-art on downstream tasks such as PubMedQA and MedMCQA dev of 77.6% and 52.9%. And despite not being trained on a general corpus, Galactica outperforms BLOOM and OPT-175B on BIG-bench. We believe these results demonstrate the potential for language models as a new interface for science. We open source the model for the benefit of the scientific community.\n\nTLDR:',
          '[START_I_SMILES]C(C(=O)O)N[END_I_SMILES]\n\n## Chemical and Physical Properties\n\nThe following are chemical properties for',
          'what is the capital of US?',
          ]

groundtruth_outputs = ["NA"] * 13

mc_output = ['Answer: (C) Generation of free radicals',
             'Answer: C Generation of free radicals',
             'Answer: C',
             'Answer: (C)',
             'A: C',
             'A: (C)',
             'Output: (C) Generation of free radicals',
             'Output: C Generation of free radicals',
             'Output: C',
             'Output: (C)',
            ]

mc_answer = ['c'] * 10

qa_output = ['Yes.',
             'Answer: Yes',
             'Answer: Yes.',
             'Yes ',
             'No.',
             'Answer: No',
             'Answer: No.',
             'No ',
             'Maybe.',
             'Answer: Maybe',
             'Answer: Maybe.',
             'Maybe ', 
            ]
qa_answer = ['yes'] * 4 + ['no'] * 4 + ['maybe'] * 4

class DataUtilsTest(unittest.TestCase):
    def test_load_data(self):
        file_name = "data/example_dataset/test/test_13.json"

        inputs, outputs, datasize = load_data(file_name=file_name)
        # Test for inputs
        for i in range(0,len(inputs)):
            self.assertEqual(inputs[i], groundtruth_inputs[i])
        # Test for outputs
        for i in range(0,len(outputs)):
            self.assertEqual(outputs[i], groundtruth_outputs[i])
        # Test for datasize
        self.assertEqual(datasize, 13)
    
    def test_batchlize(self):
        file_name = "data/example_dataset/test/test_13.json"
        inputs, outputs, datasize = load_data(file_name=file_name)
        dataset = []
        for idx in range(len(outputs)):
            dataset.append({"input":inputs[idx], "output":outputs[idx], "input_idx":idx})
        # TODO: add test for random shuffle case
        dataloader = batchlize(dataset, 4, random_shuffle= False)
        self.assertEqual(len(dataloader),  13 // 4 + 1)

    def test_answer_extraction(self):
        # Test for medmcqa dataset
        for i in range(0,len(mc_output)):
            self.assertEqual(answer_extraction(mc_output[i], answer_type="medmcqa"), mc_answer[i])
        # Test for usmle dataset
        for i in range(0,len(mc_output)):
            self.assertEqual(answer_extraction(mc_output[i], answer_type="usmle"), mc_answer[i])
        # Test for pubmedqa dataset
        for i in range(0,len(qa_output)):
            self.assertEqual(answer_extraction(qa_output[i], answer_type="pubmedqa"), qa_answer[i])
