\documentclass{article}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{float}
\usepackage{placeins}


\title{Developing an Ensemble Model for Detecting Infant Cries, Screams, and Normal Utterances}
\author{}
\date{\today}

\begin{document}

\maketitle

\section{Introduction}
The goal of this project is to develop a robust audio classification system capable of distinguishing between infant cries, screams, and normal utterances. This involves training individual models using YAMNet and Wav2Vec2 architectures, creating an ensemble of these models, and deploying the solution within a Temporal workflow.

\section{Data Acquisition and Preprocessing}
\subsection{Dataset Selection}
We utilized multiple datasets, including:
\begin{itemize}
    \item Infant Cry Audio Corpus from KAGGLE
    \item Human Screaming Detection Dataset from KAGGLE
    \item Children speech~ Audioset 4

These files have been uploaded to Google Drive and mounted to access the large dataset efficiently.
\end{itemize}

\subsection{Data Preparation}
The audio files were preprocessed to ensure consistency in format, including sampling rate and bit depth normalization. The data was segmented and labeled into three categories: `crying', `screaming', and `normal utterances'.

\section{Model Training}
\subsection{YAMNet Model}
The YAMNet model was fine-tuned for the classification task. Necessary modifications were made to adapt YAMNet for this specific application.

\subsection{Wav2Vec2 Model}
Similarly, the Wav2Vec2 model was fine-tuned to classify the audio segments effectively.

\section{Ensemble Model Development}
We combined predictions from YAMNet and Wav2Vec2 using ensemble techniques such as averaging probabilities and majority voting.

\section{Training, Testing, and Validation Approach}
\subsection{Training Approach}
To ensure robust model performance and prevent overfitting, we adopted a structured approach:
\begin{itemize}
    \item 	{Dataset Split:} The data was split into 70\% training, 15\% validation, and 15\% testing. This ensures that the models are trained on a substantial portion of the data while keeping sufficient data for validation and final testing.
    \item 	{Validation:} The validation set was used to tune hyperparameters and assess model generalization before final evaluation.
    \item 	{Testing:} The test set, containing unseen data, was used to evaluate real-world performance.
    \item 	{Data Augmentation:} Various augmentation techniques, such as noise addition and pitch shifting, were applied to increase model robustness.
    \item 	{Cross-Validation:} Employed to ensure the model generalizes well to different subsets of the dataset.
\end{itemize}

\subsection{Testing and Validation}
Model performance was evaluated using accuracy, precision, recall, and F1-score.

\section{Loss Function Justification}
We selected the {sparse categorical cross-entropy} loss function due to its suitability for multi-class classification problems. Given that our dataset consists of three distinct classes (`crying', `screaming', and `normal utterances'), this loss function is effective in handling categorical labels.

The choice of sparse categorical cross-entropy is justified as follows:
\begin{itemize}
    \item 	{Handles Multi-Class Classification Efficiently:} Since we have more than two classes, binary cross-entropy would not be appropriate. Sparse categorical cross-entropy is specifically designed for multi-class problems.
    \item 	{Computationally Efficient:} This loss function is optimized for handling integer labels without requiring one-hot encoding, reducing computational overhead.
    \item 	{Balances Experimental and Control Groups:} Our dataset contains varied samples from different sources. Sparse categorical cross-entropy ensures that all class labels contribute to the training process appropriately, preventing class imbalance from skewing the results.
    \item {Alignment with Model Architectures:} Both YAMNet and Wav2Vec2 output probability distributions over multiple categories, making categorical cross-entropy a natural fit.
\end{itemize}

\section{Performance Metrics}
We evaluated the models using:
\begin{itemize}
    \item Confusion Matrices
    \item ROC Curves
    \item Classification Reports
\end{itemize}

\section{Results and Discussion}

\subsection{YAMNet Model Results}
\begin{verbatim}
Epoch 1/10
accuracy: 0.6756 - loss: 0.9839 - val_accuracy: 0.7826 - val_loss: 0.9807
Epoch 2/10
accuracy: 0.8676 - loss: 0.6383 - val_accuracy: 0.7826 - val_loss: 0.8022
...
Epoch 10/10
accuracy: 0.8676 - loss: 0.4984 - val_accuracy: 0.7826 - val_loss: 0.7915
\end{verbatim}

\FloatBarrier 
\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{download.png}
    \caption{ROC Curve}
    \label{fig:results}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{download (1).png}
    \caption{Confusion matrix}
    \label{fig:results}
\end{figure}
\FloatBarrier 
\newpage

\subsection{Wav2Vec2 Model Results}
\begin{verbatim}
Epoch 1   Validation Loss: 0.815121
Epoch 2   Validation Loss: 0.833583
Epoch 3   Validation Loss: 0.837320
\end{verbatim}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{download (2).png}
    \caption{ROC Curve}
    \label{fig:results}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{download (3).png}
    \caption{Confusion matrix}
    \label{fig:results}
\end{figure}

\newpage

\subsection{Ensemble model}
Train Loss: 0.6878751118977865
Test Accuracy: 0.7681
Test Precision: 0.5900
Test Recall: 0.7681
Test F1 Score: 0.6674
\FloatBarrier 
\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{download (4).png}
    \caption{ROC Curve}
    \label{fig:results}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{download (5).png}
    \caption{Confusion matrix}
    \label{fig:results}
\end{figure}
\FloatBarrier 
\newpage

\subsection{Example Prediction}
\begin{verbatim}
Audio File: /content/drive/MyDrive/frontera/extracted_data/Screaming/---1_cCGK4M_out.wav
Predicted Class: [3]
\end{verbatim}
The ensemble model demonstrated improved accuracy over individual models. 

\section{Deployment with Temporal}
A Temporal workflow was designed to handle real-time audio classification with processing tasks for:
\begin{itemize}
    \item Preprocessing audio input
    \item Running ensemble classification
    \item Storing and managing results
\end{itemize}

\section{Conclusion}
This project successfully developed an ensemble model that effectively classifies infant cries, screams, and normal utterances. Future work can focus on improving real-time inference efficiency and expanding dataset diversity.
\end{document}
