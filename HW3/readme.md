The goal of this assignment is to get some experience with sequence labeling. Specifically, assigning dialogue acts to sequences of utterances in conversations from a corpus.

The raw data for each utterance in the conversation consists of the speaker name, the tokens and their part of speech tags.

The Switchboard (SWBD) corpus was collected from volunteers and consists of two person telephone conversations about predetermined topics such as child care. SWBD DAMSL refers to a set of dialogue act annotations made to this data. This [(lengthy) annotation manual](https://web.stanford.edu/~jurafsky/ws97/manual.august1.html) defines what these dialogue acts mean. In particular, see section 1c (The 42 Clustered SWBD-DAMSL Labels)

labeled data set is found in labelled data.zip
    
In all data, individual conversations are stored as individual CSV files. These CSV files have four columns and each row represents a single utterance in the conversation. The order of the utterances is the same order in which they were spoken.

```
Sample Data(0001.csv):
1    act_tag	speaker	pos	text
2    qw	A	What/WP are/VBP your/PRP$ favorite/JJ programs/NNS ?/.	What are your favorite programs? /
3    sd	B	Uh/UH ,/, it/PRP 's/BES kind/RB of/RB hard/JJ to/TO put/VB my/PRP$ finger/NN on/IN a/DT ,/, on/IN a/DT favorite/JJ T/NN V/NN program/NN ,/,	{F Uh, } it's kind of hard to put my finger [ on a, + on a ] favorite T V program, /
4    sd	B	however/RB ,/, uh/UH ,/, one/CD that/WDT I/PRP 've/VBP been/VBN watching/VBG for/IN a/DT number/NN of/IN years/NNS is/VBZ DALLAS/NNP ./.	{C however, } {F uh, } one that I've been watching for a number of years is DALLAS. /
5    sd	B	And/CC ,/, uh/UH ,/,	[ {C And, } + {F uh, }
6    ba	A	Oh/UH ,/, how/WRB funny/JJ ./.	{F Oh, } how funny. /

63   sv	A	I/PRP think/VBP that/DT is/VBZ just/RB a/DT hoot/NN ./.	I think that is just a hoot. /
```
    
```
• act_tag - the dialogue act associated with this utterance. This is blank for the unlabeled data.
• speaker - the speaker of the utterance (A or B).
• pos - a whitespace-separated list where each item is a token, "/", and a part of speech tag (e.g., "What/WP are/VBP your/PRP$ favorite/JJ programs/NNS ?/."). When the utterance has no words (e.g., the transcriber is describing some kind of noise), the pos column may be blank, consist solely of "./.", have a pos but no token, or have an invented token such as MUMBLEx. You can view the text column to see the original transcription.
• text - The transcript of the utterance with some cleanup but mostly unprocessed and untokenized. This column may or may not be a useful source of features when the utterance solely consists of some kind of noise.
```
    
- split_dataset_train_dev.py:
			partition dataset into train and dev picking files in random from directory passed as an arg , with percentage specified
      
    ```python
    (base) $ python split_dataset_train_dev.py labeled\ data 75 25
    total_no_of_files : 1076
    train_no_of_files : 807
    dev_no_of_files : 269
    Copying files into train and dev folders...
    TrainingData Dir : labeled data_807 No Of Files : 1010
    DevelopmentData Dir : labeled data_269 No Of Files : 472
    
    (base) $ python split_dataset_train_dev.py labeled\ data 93 7
    total_no_of_files : 1076
    train_no_of_files : 1000
    dev_no_of_files : 76
    Copying files into train and dev folders...
    TrainingData Dir : labeled data_1000 No Of Files : 1069
    DevelopmentData Dir : labeled data_76 No Of Files : 145
    ```
    
       labelled data was split by:
       75%-25% (labeled data_807/, labeled data_269/)
       and 
       93%-7%, (labeled data_1000/, labeled data_76/)
       so as to get 2 groups of datasets for train and development

      
- In the baseline feature set, for each utterance includes:
  - a feature for whether or not the speaker has changed in comparison with the previous utterance.
  - a feature marking the first utterance of the dialogue.
  - a feature for every token in the utterance.
  - a feature for every part of speech tag in the utterance (e.g., POS_PRP POS_RB POS_VBP POS_.).

  (FirstUtterance(1/0), SpeakerChange(1/0),
  Word tokens of utterance with prefix 'TOKEN_', Part of Speech tags of utterance with prefix 'POS_')
	    Is Convention followed to generate the following features:
	    
	    ['1', '0', 'TOKEN_What', 'TOKEN_are', 'TOKEN_your', 'TOKEN_favorite', 'TOKEN_programs', 'TOKEN_?', 'POS_WP', 'POS_VBP', 'POS_PRP$', 'POS_JJ', 'POS_NNS', 'POS_.']
	    ['0', '1', 'TOKEN_Uh', 'TOKEN_,', 'TOKEN_it', "TOKEN_'s", 'TOKEN_kind', 'TOKEN_of', 'TOKEN_hard', 'TOKEN_to', 'TOKEN_put', 'TOKEN_my', 'TOKEN_finger', 'TOKEN_on', 'TOKEN_a', 'TOKEN_,', 'TOKEN_on', 'TOKEN_a', 'TOKEN_favorite', 'TOKEN_T', 'TOKEN_V', 'TOKEN_program', 'TOKEN_,', 'POS_UH', 'POS_,', 'POS_PRP', 'POS_BES', 'POS_RB', 'POS_RB', 'POS_JJ', 'POS_TO', 'POS_VB', 'POS_PRP$', 'POS_NN', 'POS_IN', 'POS_DT', 'POS_,', 'POS_IN', 'POS_DT', 'POS_JJ', 'POS_NN', 'POS_NN', 'POS_NN', 'POS_,']
	    ['0', '0', 'TOKEN_however', 'TOKEN_,', 'TOKEN_uh', 'TOKEN_,', 'TOKEN_one', 'TOKEN_that', 'TOKEN_I', "TOKEN_'ve", 'TOKEN_been', 'TOKEN_watching', 'TOKEN_for', 'TOKEN_a', 'TOKEN_number', 'TOKEN_of', 'TOKEN_years', 'TOKEN_is', 'TOKEN_DALLAS', 'TOKEN_.', 'POS_RB', 'POS_,', 'POS_UH', 'POS_,', 'POS_CD', 'POS_WDT', 'POS_PRP', 'POS_VBP', 'POS_VBN', 'POS_VBG', 'POS_IN', 'POS_DT', 'POS_NN', 'POS_IN', 'POS_NNS', 'POS_VBZ', 'POS_NNP', 'POS_.']
	    ['0', '0', 'TOKEN_And', 'TOKEN_,', 'TOKEN_uh', 'TOKEN_,', 'POS_CC', 'POS_,', 'POS_UH', 'POS_,']
	    ['0', '1', 'TOKEN_Oh', 'TOKEN_,', 'TOKEN_how', 'TOKEN_funny', 'TOKEN_.', 'POS_UH', 'POS_,', 'POS_WRB', 'POS_JJ', 'POS_.']
	
	    ['0', '1', 'TOKEN_I', 'TOKEN_think', 'TOKEN_that', 'TOKEN_is', 'TOKEN_just', 'TOKEN_a', 'TOKEN_hoot', 'TOKEN_.', 'POS_PRP', 'POS_VBP', 'POS_DT', 'POS_VBZ', 'POS_RB', 'POS_DT', 'POS_NN', 'POS_.']

  ```python
  (base) $ python baseline_crf.py labeled\ data_807/ labeled\ data_269/ 75_25_baseline_result.txt
  Loaded Training Data
  Trained Baseline_CRF model with 100 iterations
  Predicted Sequential Labels for Test Data
  --- 200.47792506217957 seconds ---
  (base) $ python evaluate_model.py labeled\ data_269/ 75_25_baseline_result.txt 
  Overall Accuracy : 75.51742829013595
  (base) $ mv resultAnalyze.txt 75_25_baseline_resultAnalyze.txt 
  (base) $ 

  (base) $ python baseline_crf.py labeled\ data_1000/ labeled\ data_76/ 93_7_baseline_result.txt
  Loaded Training Data
  Trained Baseline_CRF model with 100 iterations
  Predicted Sequential Labels for Test Data
  --- 403.2909240722656 seconds ---
  (base) $ python evaluate_model.py labeled\ data_76/ 93_7_baseline_result.txt 
  Overall Accuracy : 75.7849594118548
  (base) $ mv resultAnalyze.txt 93_7_baseline_resultAnalyze.txt
  (base) $
  ```
  
- On inspecting *_resultAnalyze.txt of baseline features crf model it was observed the majority of sd(Statement-non-opinion) and sv(Statement-opinion) tags where getting missclassified for eachother
  
  So a feature to present a heuristic hint about sv(Statement-opinion) class was analysed:
  The following are some of sv tagged utterances :
  ```
  Well/UH then/RB ,/, you/PRP should/MD be/VB a/DT good/JJ one/CD to/TO know/VB because/IN I/PRP ,/, my/PRP$ children/NNS are/VBP grown/JJ now/RB ./.
  you/PRP should/MD
  Well/UH you/PRP can/MD always/RB get/VB up/RB and/CC leave/VB that/DT ./.
  you/PRP can/MD
  and/CC besides/IN ,/, I/PRP can/MD be/VB doing/VBG other/JJ things/NNS and/CC still/RB listen/VB to/IN the/DT news/NN ./.
  I/PRP can/MD
  And/CC you/PRP could/MD always/RB find/VB some/DT channel/NN that/WDT had/VBD something/NN on/RP
  And/CC uh/UH ,/, you/PRP could/MD always/RB catch/VB a/DT good/JJ news/NN program/NN at/IN eight/CD o'clock/RB or/CC nine/CD o'clock/RB on/IN C/NNP N/NNP N/NNP ./.
  and/CC then/RB you/PRP could/MD go/VB on/RP ,/, you/PRP know/VBP ,/, to/TO do/VB what/WP else/RB you/PRP wanted/VBD to/TO do/VB ./.
  you/PRP could/MD

  Well/UH ,/, I/PRP think/VBP the/DT public/JJ school/NN systems/NNS are/VBP doing/VBG a/DT good/JJ job/NN ./.
  I/PRP think/VBP
  I/PRP ,/, I/PRP agree/VBP with/IN you/PRP that/IN they/PRP 're/VBP starting/VBG children/NNS so/RB much/RB earlier/RBR on/IN things/NNS because/IN our/PRP$ grandchildren/NNS ,/,
  I/PRP agree/VBP

  and/CC then/RB ,/, you/PRP know/VBP ,/, the/DT special/JJ reports/NNS and/CC the/DT extended/JJ news/NN coverage/NN I/PRP thought/VBD was/VBD really/RB good/JJ ./.
  I/PRP thought/VBD
  but/CC you/PRP did/VBD n't/RB necessarily/RB have/VB to/TO watch/VB the/DT same/JJ thing/NN all/PDT the/DT time/NN ./.
  you/PRP did/VBD n't/RB

  and/CC MUMBLEx/XX ./. You/PRP know/VBP ,/, you/PRP can/MD as/RB far/RB as/IN I/PRP 'm/VBP concerned/JJ you/PRP can/MD make/VB a/DT survey/NN or/CC test/NN scores/NNS say/VB anything/NN you/PRP want/VBP it/PRP to/TO say/VB ./.
  I/PRP 'm/VBP concerned/JJ
  but/CC those/DT are/VBP ni/JJ ,/, Germantown/NNP is/VBZ a/DT nice/JJ area/NN ./.
  It/PRP 's/BES a/DT nice/JJ area/NN ./.
  nice/JJ area/NN
  ```
      
  ```
  posPairSet = {('PRP', 'MD'), ('PRP', 'VBP'), ('PRP', 'VBD')} for such occurence sv_True was enabled
  ['0', '1', 'sv_True', 'TOKEN_I', 'TOKEN_think', 'TOKEN_that', 'TOKEN_is', 'TOKEN_just', 'TOKEN_a', 'TOKEN_hoot', 'TOKEN_.',
                        'POS_PRP', 'POS_VBP',     'POS_DT',     'POS_VBZ',  'POS_RB',     'POS_DT',  'POS_NN', 'POS_.']
  ['0', '0', 'sv_True', 'sv_True', 'TOKEN_So', 'TOKEN_I', 'TOKEN_think', 'TOKEN_that', 'TOKEN_,', 'TOKEN_uh', 'TOKEN_,', 'TOKEN_they', 'TOKEN_need', 'TOKEN_to', 'TOKEN_look', 'TOKEN_into', 'TOKEN_it', 
                                   'POS_RB',   'POS_PRP', 'POS_VBP',      'POS_IN',     'POS_,',   'POS_UH', 'POS_,',    'POS_PRP', 'POS_VBP',        'POS_TO', 'POS_VB', 'POS_IN', 'POS_PRP']
  ```
      
  ```python
  (base) $ python advanced_crf.py labeled\ data_807/ labeled\ data_269/ 75_25_advanced_svTrue_result.txt
  Loaded Training Data
  Trained Advanced_CRF model with 100 iterations
  Predicted Sequential Labels for Test Data
  --- 274.95387601852417 seconds ---
  (base) $ python evaluate_model.py labeled\ data_269/ 75_25_advanced_svTrue_result.txt
  Overall Accuracy : 80.34159669907854
  (base) $ mv resultAnalyze.txt 75_25_advanced_svTrue_resultAnalyze.txt 

  (base) $ python advanced_crf.py labeled\ data_1000/ labeled\ data_76/ 93_7_advanced_svTrue_result.txt
  Loaded Training Data
  Trained Advanced_CRF model with 100 iterations
  Predicted Sequential Labels for Test Data
  --- 476.31964898109436 seconds ---
  (base) $ python evaluate_model.py labeled\ data_76/ 93_7_advanced_svTrue_result.txt
  Overall Accuracy : 80.29560422729361
  (base) $ mv resultAnalyze.txt 93_7_advanced_svTrue_resultAnalyze.txt 
  (base) $ 
  ```
      
  Hence with sv_True feature, we were able to reduce no of misclassification errors of '_dssv'
  
  Accuracy Gain (compared to baseline features): ~75% - ~80% = ~ +5%

- Some heuristic to indicate what utterance constitutes a question(based on question act_tags) and 
	response would be followed by a speaker change
	questionTagSet = {'qy','qw','qy^d','bh','qo', 'qh', '^g', 'qw^d'}
	on speaker change if previous one is a question current should be a response

  ```
	qy	['Question', '0', '1', 'TOKEN_Is', 'TOKEN_that', 'TOKEN_,', 'POS_VBZ', 'POS_DT', 'POS_,']
	qy	['Question', '0', '0', 'TOKEN_that', 'TOKEN_is', 'TOKEN_right', 'TOKEN_,', 'TOKEN_is', "TOKEN_n't", 'TOKEN_it', 'TOKEN_?', 'POS_DT', 'POS_VBZ', 'POS_JJ', 'POS_,', 'POS_VBZ', 'POS_RB', 'POS_PRP', 'POS_.']
	ny	['Response', '0', '1', 'TOKEN_Yeah', 'TOKEN_.', 'POS_UH', 'POS_.']
	sd	['0', '1', 'TOKEN_Because', 'TOKEN_they', 'TOKEN_ask', 'TOKEN_me', 'TOKEN_,', 'TOKEN_why', 'TOKEN_are', 'TOKEN_there', 'TOKEN_two', 'TOKEN_bridges', 'TOKEN_going', 'TOKEN_into', 'TOKEN_Dallas', 'TOKEN_?', 'POS_IN', 'POS_PRP', 'POS_VBP', 'POS_PRP', 'POS_,', 'POS_WRB', 'POS_VBP', 'POS_EX', 'POS_CD', 'POS_NNS', 'POS_VBG', 'POS_IN', 'POS_NNP', 'POS_.']
  ```

  ```python
  (base) $ 
  (base) $ python advanced_crf.py labeled\ data_807/ labeled\ data_269/ 75_25_advanced_QuestionResponse_result.txt
  Loaded Training Data
  Trained Advanced_CRF model with 100 iterations
  Predicted Sequential Labels for Test Data
  --- 515.9918510913849 seconds ---
  (base) $ python evaluate_model.py labeled\ data_269/ 75_25_advanced_QuestionResponse_result.txt
  Overall Accuracy : 76.98425867542782
  (base) $ mv resultAnalyze.txt 75_25_advanced_QuestionResponse_resultAnalyze.txt

  (base) $ python advanced_crf.py labeled\ data_1000/ labeled\ data_76/ 93_7_advanced_QuestionResponse_result.txt
  Loaded Training Data
  Trained Advanced_CRF model with 100 iterations
  Predicted Sequential Labels for Test Data
  --- 439.0546200275421 seconds ---
  (base) $ python evaluate_model.py labeled\ data_76/ 93_7_advanced_QuestionResponse_result.txt
  Overall Accuracy : 77.24766426711595
  (base) $ mv resultAnalyze.txt 93_7_advanced_QuestionResponse_resultAnalyze.txt
  (base) $ 
  ```
  
- Context info with act_tag labels of previous 2 utterances(expect first 2 of a conversation)

  ```
	qw	['1', '0', 'TOKEN_What', 'TOKEN_are', 'TOKEN_your', 'TOKEN_favorite', 'TOKEN_programs', 'TOKEN_?', 'POS_WP', 'POS_VBP', 'POS_PRP$', 'POS_JJ', 'POS_NNS', 'POS_.']
	sd	['0', '1', 'TOKEN_Uh', 'TOKEN_,', 'TOKEN_it', "TOKEN_'s", 'TOKEN_kind', 'TOKEN_of', 'TOKEN_hard', 'TOKEN_to', 'TOKEN_put', 'TOKEN_my', 'TOKEN_finger', 'TOKEN_on', 'TOKEN_a', 'TOKEN_,', 'TOKEN_on', 'TOKEN_a', 'TOKEN_favorite', 'TOKEN_T', 'TOKEN_V', 'TOKEN_program', 'TOKEN_,', 'POS_UH', 'POS_,', 'POS_PRP', 'POS_BES', 'POS_RB', 'POS_RB', 'POS_JJ', 'POS_TO', 'POS_VB', 'POS_PRP$', 'POS_NN', 'POS_IN', 'POS_DT', 'POS_,', 'POS_IN', 'POS_DT', 'POS_JJ', 'POS_NN', 'POS_NN', 'POS_NN', 'POS_,']
	sd	['0', '0', 'PREV_TO_PREV_qw', 'PREV_sd', 'TOKEN_however', 'TOKEN_,', 'TOKEN_uh', 'TOKEN_,', 'TOKEN_one', 'TOKEN_that', 'TOKEN_I', "TOKEN_'ve", 'TOKEN_been', 'TOKEN_watching', 'TOKEN_for', 'TOKEN_a', 'TOKEN_number', 'TOKEN_of', 'TOKEN_years', 'TOKEN_is', 'TOKEN_DALLAS', 'TOKEN_.', 'POS_RB', 'POS_,', 'POS_UH', 'POS_,', 'POS_CD', 'POS_WDT', 'POS_PRP', 'POS_VBP', 'POS_VBN', 'POS_VBG', 'POS_IN', 'POS_DT', 'POS_NN', 'POS_IN', 'POS_NNS', 'POS_VBZ', 'POS_NNP', 'POS_.']
	sd	['0', '0', 'PREV_TO_PREV_sd', 'PREV_sd', 'TOKEN_And', 'TOKEN_,', 'TOKEN_uh', 'TOKEN_,', 'POS_CC', 'POS_,', 'POS_UH', 'POS_,']
	ba	['0', '1', 'PREV_TO_PREV_sd', 'PREV_sd', 'TOKEN_Oh', 'TOKEN_,', 'TOKEN_how', 'TOKEN_funny', 'TOKEN_.', 'POS_UH', 'POS_,', 'POS_WRB', 'POS_JJ', 'POS_.']
	+	['0', '1', 'PREV_TO_PREV_sd', 'PREV_ba', 'TOKEN_And', 'TOKEN_,', 'TOKEN_uh', 'TOKEN_,', 'TOKEN_it', "TOKEN_'s", 'TOKEN_going', 'TOKEN_to', 'TOKEN_be', 'TOKEN_going', 'TOKEN_off', 'TOKEN_the', 'TOKEN_air', 'TOKEN_,', 'TOKEN_uh', 'TOKEN_,', 'TOKEN_let', "TOKEN_'s", 'TOKEN_see', 'TOKEN_,', 'TOKEN_a', 'TOKEN_week', 'TOKEN_from', 'TOKEN_Fri-', 'TOKEN_,', 'TOKEN_a', 'TOKEN_week', 'TOKEN_from', 'TOKEN_tonight', 'TOKEN_.', 'POS_CC', 'POS_,', 'POS_UH', 'POS_,', 'POS_PRP', 'POS_BES', 'POS_VBG', 'POS_TO', 'POS_VB', 'POS_VBG', 'POS_IN', 'POS_DT', 'POS_NN', 'POS_,', 'POS_UH', 'POS_,', 'POS_VB', 'POS_PRP', 'POS_VB', 'POS_,', 'POS_DT', 'POS_NN', 'POS_IN', 'POS_NNP', 'POS_,', 'POS_DT', 'POS_NN', 'POS_IN', 'POS_RB', 'POS_.']
  ```
  
  ```python
  (base) $ python advanced_crf.py labeled\ data_807/ labeled\ data_269/ 75_25_advanced_prevContext_result.txt
  Loaded Training Data
  Trained Advanced_CRF model with 100 iterations
  Predicted Sequential Labels for Test Data
  --- 227.58054900169373 seconds ---
  (base) $ python evaluate_model.py labeled\ data_269/ 75_25_advanced_prevContext_result.txt
  Overall Accuracy : 79.20552218497993
  (base) $ mv resultAnalyze.txt 75_25_advanced_prevContext_resultAnalyze.txt

  (base) $ python advanced_crf.py labeled\ data_1000/ labeled\ data_76/ 93_7_advanced_prevContext_result.txt
  Loaded Training Data
  Trained Advanced_CRF model with 100 iterations
  Predicted Sequential Labels for Test Data
  --- 167.78879618644714 seconds ---
  (base) $ python evaluate_model.py labeled\ data_76/ 93_7_advanced_prevContext_result.txt
  Overall Accuracy : 79.86674835349977
  (base) $ mv resultAnalyze.txt 93_7_advanced_prevContext_resultAnalyze.txt
  (base) $ 
  ```
  
- Combination of all advanced feature set:

  ((Question/Response)if_applicable,
  FirstUtterance(1/0), SpeakerChange(1/0),
  (PREV_TO_PREV_act_tag, PREV_act_tag) for all utterances except first 2, (indicator for sv_tag)if applicable,
  Word tokens of utterance with prefix 'TOKEN_', Part of Speech tags of utterance with prefix 'POS_')
	    
  ```
  ['Question', '1', '0', 'TOKEN_What', 'TOKEN_are', 'TOKEN_your', 'TOKEN_favorite', 'TOKEN_programs', 'TOKEN_?', 'POS_WP', 'POS_VBP', 'POS_PRP$', 'POS_JJ', 'POS_NNS', 'POS_.']
  ['Response', '0', '1', 'TOKEN_Uh', 'TOKEN_,', 'TOKEN_it', "TOKEN_'s", 'TOKEN_kind', 'TOKEN_of', 'TOKEN_hard', 'TOKEN_to', 'TOKEN_put', 'TOKEN_my', 'TOKEN_finger', 'TOKEN_on', 'TOKEN_a', 'TOKEN_,', 'TOKEN_on', 'TOKEN_a', 'TOKEN_favorite', 'TOKEN_T', 'TOKEN_V', 'TOKEN_program', 'TOKEN_,', 'POS_UH', 'POS_,', 'POS_PRP', 'POS_BES', 'POS_RB', 'POS_RB', 'POS_JJ', 'POS_TO', 'POS_VB', 'POS_PRP$', 'POS_NN', 'POS_IN', 'POS_DT', 'POS_,', 'POS_IN', 'POS_DT', 'POS_JJ', 'POS_NN', 'POS_NN', 'POS_NN', 'POS_,']
  ['0', '0', 'PREV_TO_PREV_qw', 'PREV_sd', 'TOKEN_however', 'TOKEN_,', 'TOKEN_uh', 'TOKEN_,', 'TOKEN_one', 'TOKEN_that', 'TOKEN_I', "TOKEN_'ve", 'TOKEN_been', 'TOKEN_watching', 'TOKEN_for', 'TOKEN_a', 'TOKEN_number', 'TOKEN_of', 'TOKEN_years', 'TOKEN_is', 'TOKEN_DALLAS', 'TOKEN_.', 'POS_RB', 'POS_,', 'POS_UH', 'POS_,', 'POS_CD', 'POS_WDT', 'POS_PRP', 'POS_VBP', 'POS_VBN', 'POS_VBG', 'POS_IN', 'POS_DT', 'POS_NN', 'POS_IN', 'POS_NNS', 'POS_VBZ', 'POS_NNP', 'POS_.']
  ['0', '0', 'PREV_TO_PREV_sd', 'PREV_sd', 'TOKEN_And', 'TOKEN_,', 'TOKEN_uh', 'TOKEN_,', 'POS_CC', 'POS_,', 'POS_UH', 'POS_,']
  ['0', '1', 'PREV_TO_PREV_sd', 'PREV_sd', 'TOKEN_Oh', 'TOKEN_,', 'TOKEN_how', 'TOKEN_funny', 'TOKEN_.', 'POS_UH', 'POS_,', 'POS_WRB', 'POS_JJ', 'POS_.']

  ['0', '1', 'PREV_TO_PREV_sv', 'PREV_sd', 'sv_True', 'TOKEN_I', 'TOKEN_think', 'TOKEN_that', 'TOKEN_is', 'TOKEN_just', 'TOKEN_a', 'TOKEN_hoot', 'TOKEN_.', 'POS_PRP', 'POS_VBP', 'POS_DT', 'POS_VBZ', 'POS_RB', 'POS_DT', 'POS_NN', 'POS_.']
  ```
  
  ```python
  (base) $ python advanced_crf.py labeled\ data_807/ labeled\ data_269/ 75_25_advanced_all_result.txt
  Loaded Training Data
  Trained Advanced_CRF model with 100 iterations
  Predicted Sequential Labels for Test Data
  --- 390.81125807762146 seconds ---
  (base) $ python evaluate_model.py labeled\ data_269/ 75_25_advanced_all_result.txt
  Overall Accuracy : 83.8925209349661
  (base) $ mv resultAnalyze.txt 75_25_advanced_all_resultAnalyze.txt
  (base) $ 

  (base) $ python advanced_crf.py labeled\ data_1000/ labeled\ data_76/ 93_7_advanced_all_result.txt
  Loaded Training Data
  Trained Advanced_CRF model with 100 iterations
  Predicted Sequential Labels for Test Data
  --- 419.14071822166443 seconds ---
  (base) $ python evaluate_model.py labeled\ data_76/ 93_7_advanced_all_result.txt
  Overall Accuracy : 84.03277684178282
  (base) $ mv resultAnalyze.txt 93_7_advanced_all_resultAnalyze.txt
  (base) $ 
  ```

- CRF model was built using default L-BFGS training algorithm with Elastic Net (L1 + L2) regularization,as conveyed in [reference](http://nbviewer.jupyter.org/github/tpeng/python-crfsuite/blob/master/examples/CoNLL%202002.ipynb). 
  - Much focus was spent on designing advanced NLP features which could aid in sequence labelling, keeping the rest of hyperparameters of algorithm intact
  - No of training iterations was varied to get both baseline and advanced crf models to be trained within 600 sec.

- evaluate_model.py : To evaluate models on the development data, that compares predicted labels to the actual labels of the utterances in the CSV files and calculates accuracy 
