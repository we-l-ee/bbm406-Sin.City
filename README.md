# Songs in the Noisy City

## Abstract
In most cultures we can define a generic type of songs that defines the culture. We can even pinpoint the differences between similar cultures but relatively in a different region which might be city, county or state. This is the point where idea came to classify folk songs in Turkey region by region. Turkey has 7 geographic regions which can be seen as cultural regions. Thus, those regions do not have completely different cultures,but differences are visible which one of them is the regional songs. For instance, tulum, kemençe(endemic musical instruments for Karadeniz) can be heard in Karadeniz songs which are distinguishable from other regions songs. However, some regions have same musical infrastructure which are rhythm, instruments, chords or notes patterns and vocal type which is hard to distinguish for a human being. This one is probably the hardest challenge we will face. Hence, feature extraction is the crucial point for this matter and there are certain features can be used for our classification problem and there are some must be tried before used. We will try to implement and analyze different models and different features for this classification problem.

## Introduction 
In machine learning most important step is the feature extraction. Just by improving features will increase dramatically for contemporary machine learning models. In the audio feature extraction, current state-of-art is being provided by MIR(Music Information Retrieval) research field. MIR is a huge field that has been under the spotlight since Tzanetakis and Cook released a paper on Music Genre Classification in 2002[1]. MIR includes audio feature extraction, signal processing, audio analyse etc. Audio feature extraction is what we are interested in MIR research field. We are planning mainly to use features: Chroma-gram, MFCC(Mel-frequency cepstral coefficients), Spectral centroid, Spectral roll-of, Zero-crossing rate, etc. We will also use other features for reference points but these will be the main features. 

Machine learning model selection for the job is not without importance. State-of-art models are plentiful, however they are mostly specific for a given problem. In order to find the best classification, we will use various models which will be CNN(Convolutional Neural Network), SVM(Support Vector Machine), K-NN(K Nearest Neighbor). These machine learning models will be implemented with various contemporary features discussed in MIR. Subsequently these will be analysed.   
Analyzing will be last step in our paper. We will gather results after the testing which in a matrix as rows for machine learning models and columns for features. Best model and features will be apparent after the testing is done. Confusion matrix will be constructed by using our best model and features.

## Related Works
In this paper, classification problem is more specific than any other work. since Tzanetakis and Cook presented music genre classification[1], a lot of work has been done. 
For Audio processing and analyzing, it is hard to build-up a reliable feature extractor system for audio seems to be a more challenging task than classification problem.
For this purpose some researches have been done about features extraction such as; MFCC[4], Chromogram[5], Zero-crossing rate[6], Spectral-Centroid[7].

However,most known study for folk song classification which is based on region has been done for China[3]. On paper in interest, 74 features have been extracted from audio files of the songs, and classified by an audio classifier on SVM which is one of our concern. Research shows usthat SVM without feature selection is a very effective classification method with the combination of 13-dimension MFCC and 10-dimension without feature selection[3]. It points us to post processing and segmentation plays key role in here. Other research is called "Regional Classification of Traditional Japanese Folk Songs" by Akihiro KAWASE, Akifumi TOKOSUMI has been made. That research use music corpora consisting of 202,246 tones from 1,794 song pieces from 45 prefectures in Japan which is big enough.



## Methodology 
### Data Set 
We have gathered our data from youtube playlist. There are various regional playlist which we have found with quick search[2]. We obtained these in ogg format, smallest bitrate we can find in order to save space. JDownloader[3] used to download these playlist with ease. Each download placed in respective folder for label in their classification. In these playlist there might be noise such as given song might not be in that regional as suggested in the name of the playlist.




## Feature Works

## Reference Link
1. [Music Genre Classification, Tzanetakis and Cook](http://dspace.library.uvic.ca:8080/bitstream/handle/1828/1344/tsap02gtzan.pdf?sequence=1)
2. [An evaluation of Convolutional Neural Networks for music
classification using spectrograms, Yandre M.G. Costaa, Luiz S. Oliveira b, Carlos N. Silla Jr. c]http://www.inf.ufpr.br/lesoliveira/download/ASOC2017.pdf
3. [The Study of the Classification of Chinese Folk Songs by Regional Style, Yi Liu, JiePing Xu, Lei Wei, Yun Tian]http://ieeexplore.ieee.org/abstract/document/4338407/
4. [Design, analysis and experimental evaluation of block based transformation in MFCC computation for speaker recognition, Md.Sahidullah, Goutam Saha]http://www.sciencedirect.com/science/article/pii/S0167639311001622
5. [CLASSIFYING MUSIC AUDIO
WITH TIMBRAL AND CHROMA FEATURES, Daniel P. W. Ellis]https://www.ee.columbia.edu/~dpwe/pubs/Ellis07-timbrechroma.pdf
6.[Improving Music Genre Classification by Short Time Feature Integration, Meng, Anders, Ahrendt, Peter,  Larsen, Jan] http://orbit.dtu.dk/en/publications/improving-music-genre-classification-by-short-time-feature-integration(551fef78-45e4-43bf-bae4-b2677b1f10fd).html
7.[Spectral centroid and timbre in complex, multiple instrumental textures, Emery Schubert, Joe Wolfe, Alex Tarnopolsky] https://www.google.com.tr/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=0ahUKEwir26u9mPjXAhVBL1AKHamvD6YQFggoMAA&url=https%3A%2F%2Fwww.researchgate.net%2Fpublication%2F200806323_Spectral_centroid_and_timbre_in_complex_multiple_instrumental_textures&usg=AOvVaw0zM409_UuYOW9nHoneIsvm
22. [Data Sets](https://we-l-ee.github.io/bbm406-Sin.City/#dataset-links)
23. [JDownloader](http://jdownloader.org/)

# Related Works
* https://pdfs.semanticscholar.org/c11a/c956b26df3df4c2c6a4eda097b4e1cfbdb4f.pdf
* http://www.ifs.tuwien.ac.at/~andi/publications/pdf/lid_ismir05.pdf
* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4676707/ - automatic !
* http://josh-jacobson.github.io/genre-classification/doc/paper.pdf 
* https://www.researchgate.net/publication/319326354_Music_Feature_Maps_with_Convolutional_Neural_Networks_for_Music_Genre_Classification - Convulutional Neural Network
* http://www.terasoft.com.tw/conf/ismir2014/LBD%5CLBD17.pdf - CNN
* http://cs231n.stanford.edu/reports/2017/pdfs/22.pdf - Deep Learning
* http://www.inf.ufpr.br/lesoliveira/download/ASOC2017.pdf - CNN
* http://www.iaeng.org/publication/IMECS2010/IMECS2010_pp546-550.pdf


# Dataset Links
1. Karadeniz 
	* https://www.youtube.com/watch?v=MUykuOSaDjw&list=PLNLFBIgt47CQNAhy8XDB6gU2ox-_gxK_R
	* https://www.youtube.com/watch?v=VH6Km2swSy0&list=PLYSWkD-kyTdpLOK1aVbbEwNyQSS1rW36S
2. Rumeli-Trakya
	* https://www.youtube.com/watch?v=bQYU3ZDH4KQ&list=PL6323BD7D858BF0ED
	* https://www.youtube.com/watch?v=AFywjUmeDEQ&list=PL00513E5A6FFC0499
	* https://www.youtube.com/watch?v=eQiv8Aq3CXA&list=PL9CD15E7677AAF4C7
3. Ege
	* https://www.youtube.com/watch?v=rWGPjMWj1kw&list=PLgtKoGJSRR0lncZvBCiSGoYRAtUpSbwTr
	* https://www.youtube.com/watch?v=6ruKlB_9XYk&list=PL0C40467C2CE43B89
4. Akdeniz-Kıbrıs
	* https://www.youtube.com/watch?v=r6e1IpOUG_k&list=PLayG2GW7GMIVYja5AUY3bQqV7zBOOHhsC
	* https://www.youtube.com/watch?v=ylgnkW5ShZM&list=PLLIjmHg_D4HBqeLV3fH6yR5cDttvKlGzp
	* https://www.youtube.com/watch?v=sFdCjXoDWKs
	* https://www.youtube.com/watch?v=vl0sxDiEeJ8&list=PLExdayjZ9-fCAsOzTgGz-QAkoo0lrkdB7
	* https://www.youtube.com/watch?v=NLiwg55R1m8&list=PLMnJx5pto0BgInhnOuBrx8Zq5Xjw16_Hs
	* https://www.youtube.com/watch?v=HJNyn1zvKxc&list=PLNJYACUdYVca0uyCpbYMmhEPAvL3_Blz6
	* https://www.youtube.com/watch?v=1YX9RJyoZIg&list=PL0KzykfjtmASTPjaBJiJEANcRapSfrokH
5. İç Anadolu
	* https://www.youtube.com/watch?v=6g37P3aOMkA&list=RDQM049j_kqtnow 
	* https://www.youtube.com/watch?v=0UyLho-VXx4&list=PL6J_woZ6ntHc0q2_MrTlzvdyQzyX2a7Yp
6. Doğu Anadolu - Kafkas
	* https://www.youtube.com/watch?v=p_68Pnk1kKI&list=PLA3741D504BE94E02
	* https://www.youtube.com/watch?v=N2a_kWyalac&list=PL7G00IBaD4iqkXVPMgpPy0eapH2yDjIO_
	* https://www.youtube.com/watch?v=Np2QrvEIe6w&list=RDQMI7Fmbi3y8PA
	* https://www.youtube.com/watch?v=Dap_clUjNRM&list=PL4-s2DxLiFFYUg7xYHzGInflfSXQKAieO
	* https://www.youtube.com/watch?v=wC8xHJ0dQBg&list=PLe93eZ82udlDIOnYArSz1xrb09OxYnBes
7. Güney Doğu Anadolu
	* https://www.youtube.com/watch?v=rQWSGoFePSQ&list=PLsV8dtj0vUL4bc-PSFPqDe_DQdTs8N4bM
	* https://www.youtube.com/watch?v=V_bS79ADL5k&list=PLDF3BEF0BF738F4E3
	* https://www.youtube.com/watch?v=GRGJuQE-cs8&list=PLD0QrDbeHEyXwseQVFd50oF92-M8IQEuX
	* https://www.youtube.com/watch?v=rZBDVfT94Ho&list=PLe93eZ82udlAB6co3p2bN4LzaEaIUG-CV

# Feature Extraction and Libraries 	
* Library Comparasion Paper: https://www.ntnu.edu/documents/1001201110/1266017954/DAFx-15_submission_43_v2.pdf/06508f48-9272-41c8-9381-7639a0240770
* Libraries
	* libXtract: http://libxtract.sourceforge.net/
	* Yaafe: http://yaafe.sourceforge.net/
	* Marsyas: http://marsyas.info/index.html
	* librosa: https://librosa.github.io/
* Useful Links For Libraries
	* https://docs.python.org/2/library/wave.html
	* http://rwx.io/blog/2016/04/08/bp-pyaudioanalysis/
	* https://github.com/worldveil/dejavu
	* https://stackoverflow.com/questions/20719558/feature-extraction-from-an-audio-file-using-python
* Music Information Retrieval
	* http://musicinformationretrieval.com/ - good to start for definitions
	* https://www.researchgate.net/publication/221787719_Machine_Learning_Approaches_for_Music_Information_Retrieval
	* http://www.ee.columbia.edu/~dpwe/pubs/MandelE08-MImusic.pdf
	* http://musicweb.ucsd.edu/~sdubnov/Mu270d/DeepLearning/FeaturesAudioEck.pdf - Deep Belief Networks
* FEATURES
	* Mel-frequency cepstral coefficients (MFCC) - the coefficients that collectively make up the short-term power spectrum of a sound
	* Mel-scaled power spectrogram - the Mel Scale is used to provide greater resolution for more informative (lower) frequencies
	* Chromagram of a short-time Fourier transform - projects into bins representing the 12 distinct semitones (or chroma) of the 			musical octave
	* Octave-based spectral contrast - distributions of sound energy over octave frequencies
	* Tonnetz - estimates tonal centroids as coordinates in a six-dimensional interval space
	* Centroid
	* Features in the http://josh-jacobson.github.io/genre-classification/ might be useful
	* Features in the http://www.ifs.tuwien.ac.at/~andi/publications/pdf/lid_ismir05.pdf different than the others
	* http://www.ifs.tuwien.ac.at/~schindler/lectures/IR_VU_SS2010-MIR_pt2-TL%20(Alex)%202013.pdf - useful
	* 
* For Better Features
	* http://deepsound.io/music_genre_recognition.html
	* PCA, Dimension Reduction, Feature Selection, Discrimination, LDA
	
# Related Projects
* https://github.com/meetshah1995/crnn-music-genre-classification 
* https://github.com/jsalbert/Music-Genre-Classification-with-Deep-Learning
* https://github.com/mlachmish/MusicGenreClassification

# Machine Learning Libraries Options
* TensorFlow
* scikit-learn
* Theano or pylearn2 which is written on top of Theano
