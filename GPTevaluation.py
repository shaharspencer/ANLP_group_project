import csv
import sys
import os
import re
os.environ["OPENAI_API_KEY"] = "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"
# from langchain import PromptTemplate
# from langchain.chat_models import ChatOpenAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.docstore.document import Document


def evaluate_with_gpt(introduction, abstract):
    return "cool!"
#     llm = ChatOpenAI(model_name=model_name)
    prompt_template = "I will now give you an introduction of a scientific paper and an abstract proposal written based on this introduction. Please rate how well the abstract sumarizes the article, based on the following parameters:" \
                      "\n1. Relevence - whether the main points of the paper described in the introduction appear in the abstract. Provide a score between 0 (no details in the abstract) to 100 (all the introduction details are in the abstract)." \
                      "\n2. Accuracy - whether there are any details that appear in the abstract but not in the introduction. Provide a score between 0 (there are many details in the abstract that do not appear in the introduction) to 100 (there are no extra details in the abstract)." \
                      "\nYour output should only be the score in exactly this format: (Relevence score,Accuracy score).\n" \
                      "Here is the introduction:\n" \
                      "{context}\n" \
                      "Here is the proposed abstract:\n" \
                      "{question}\n" \
                      "Your answer:"
#     PROMPT = PromptTemplate(
#         template=prompt_template, input_variables=["context", "question"]
#     )
#     chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)
#     result = chain({"input_documents": [introduction], "question": abstract}, return_only_outputs=True)
#     try:
#         results = re.search(r"\(\s*([^,]+)\s*,\s*([^)]+)\s*\)", result['output_text'])
#         f1_score = 2*float(results.group(1))*float(results.group(2))/(float(results.group(1))+float(results.group(2)))
#         final_result = {"Relevence (precision) score": float(results.group(1)), "Accuracy (recall) score": float(results.group(2)), "F1 score": f1_score}
#     except:
#         final_result = result
#     return final_result


# def usage_example(): #here is a usage example
#     model_name = 'gpt-3.5-turbo-16k'
#     text = "The recent success of deep neural networks (DNNs) for vi sion tasks owes much to the availability of largescale, cor rectly annotated datasets. However, obtaining such high quality datasets can be extremely expensive, and sometimes even impossible. The common approaches, such as web queries [Liet al. , 2017 ]and crowdsourcing [Song et al. , 2019 ], can easily provide extensive labeled data, but unavoid ably introduce noisy labels . Existing studies [Arpit et al. , 2017; Zhang et al. , 2021 ]have demonstrated that DNNs can easily overﬁt noisy labels, which deteriorates the generaliza tion performance. Thus, it is essential to develop noiserobust algorithms for learning with noisy labels. Given a noisy training set consisting of clean samples and mislabeled samples, a common category of approaches [Reed et al. , 2015; Arazo et al. , 2019; Zhang et al. , 2020 ]to mit igating the negative inﬂuence of noisy labels is to identify and correct the mislabeled samples. However, the correction procedure in these methods only updates the noisy labels us ing the model prediction from the most recent training epoch directly, thus it may suffer from the false correction as themodel predictions for noisy samples tend to ﬂuctuate. Take a bird image mislabeled as an airplane as an example. Dur ing the training, the clean bird samples would encourage the model to predict a given bird image as a bird, while the bird images with airplane labels regularly pull the model back to predict the bird as an airplane. Hence, the model prediction gathered in one training epoch may change back and forth between bird and airplane, resulting in false correction. We investigate the reason for performance degradation by analyzing the memorization behavior of the DNNs models. We observe that there exists a turning point during training. Before the turning point, the model only learns from easy (clean) samples, and thus model prediction is likely to be con sistent with clean samples. After the turning point, the model increasingly memorizes hard (mislabeled) samples. Hence model prediction oscillates strongly on clean samples. Trig gered by this observation, we seek to make the model retain the earlylearning memory for consistent predictions on clean samples even after the turning point. In this paper, we propose selfensemble label correction (SELC), which potentially corrects noisy labels during train ing thus preventing the model from being affected by the noisy labels. SELC leverages the knowledge provided in the model predictions over historical training epochs to form a consensus of prediction (ensemble prediction) before the turning point. We demonstrate that combining ensemble pre diction with the original noisy label leads to a better target. Accordingly, the model is gradually reﬁned as the targets be come less noisy, resulting in improving performance. How ever, it is challenging to ﬁnd the turning point. Existing works estimate the turning point based on a test set or noise infor mation, which are unobservable in practice. We propose a metric to estimate the turning point only using training data, allowing us to select a suitable initial epoch to perform SELC. Overall, our contributions are summarized as follows: • We propose a simple and effective label correction method SELC based on selfensembling. • We design an effective metric based on unsupervised loss modeling to detect the turning point without requir ing the test set and noise information. • SELC achieves superior results and can be integrated with other techniques such as mixup [Zhang et al. , 2018 ] to further enhance the performance.arXiv:2205.01156v1  [cs.CV]  2 May 2022(a) Training accuracy  (b) Test accuracy  (c) CE  (d) CE  (e) SELC  (f) SELC Figure 1: Plots (a) and (b) show the training and test accuracy on CIFAR10 with different ratios of label noise using crossentropy (CE) loss. We investigate the memorization behavior of DNNs on CIFAR10 with 60% label noise using CE loss and SELC. Plot (c) and (e) show the fraction of clean samples that are predicted correctly (blue) and incorrectly (black). Plot (d) and (f) show the fraction of mislabeled samples that are predicted correctly (blue), memorized (i.e. the prediction equals to the wrong label, shown in red), and incorrectly predicted as neither the true nor the given wrong label (black). Compared to CE, SELC effectively prevents memorization of mislabeled samples and reﬁnes the model to attain correct predictions on both clean and mislabeled samples. 2 Related Work "
#     abstract = "Deep neural networks are prone to overfitting noisy labels, resulting in poor generalization performance. To overcome this problem, we present a simple and effective method self-ensemble label correction (SELC) to progressively correct noisy labels and refine the model. We look deeper into the memorization behavior in training with noisy labels and observe that the network outputs are reliable in the early stage. To retain this reliable knowledge, SELC uses ensemble predictions formed by an exponential moving average of network outputs to update the original noisy labels. We show that training with SELC refines the model by gradually reducing supervision from noisy labels and increasing supervision from ensemble predictions. Despite its simplicity, compared with many state-of-the-art methods, SELC obtains more promising and stable results in the presence of class-conditional, instance-dependent, and real-world label noise. The code is available at https://github.com/MacLLL/SELC."
#     text_document = Document(page_content=text)
#     gpt_evaluation = evaluate_with_gpt(text_document, abstract, model_name)
#     print(gpt_evaluation)


def process_pairs(input_csv, input_txt, output_csv, processing_function):
    with open(input_csv, 'r', newline='', encoding='utf-8') as csv_file, \
            open(input_txt, 'r', encoding='utf-8') as txt_file, \
            open(output_csv, 'w', newline='', encoding='utf-8') as output_file:
        csv_reader = csv.reader(csv_file)
        txt_lines = txt_file.readlines()
        header = next(csv_reader)  # Skip the header

        csv_writer = csv.writer(output_file)
        csv_writer.writerow(
            header + ['prediction'] + ['result'])  # Add a new column for results

        for csv_row, txt_line in zip(csv_reader, txt_lines):
            introduction = csv_row[-1]
            prediction = txt_line.strip()
            result = processing_function(introduction, prediction)
            csv_writer.writerow(csv_row + [prediction] + [result])

def main():
    dir = "data_files/routing_protocols"
    input_csv_file = os.path.join(dir,'abstracts_introductions_test_split.csv')
    input_txt_file = os.path.join(dir,'generated_predictions_TRAIN_protocols_TEST_protocols.txt')
    output_csv_file = os.path.join(dir,'FOR_GPT_generated_predictions_TRAIN_protocols_TEST_protocols.csv')

    # Call the method with your processing function
    process_pairs(input_csv_file, input_txt_file, output_csv_file, evaluate_with_gpt)



if __name__ == "__main__":
    main()
