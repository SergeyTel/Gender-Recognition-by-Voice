# Классификация пола человека по аудиозаписи голоса
Для записи собственных данных использовался язык R с библиотекой warbleR
#### Команды для обработки wav файлов на языке R:
1. dataframe <- data.frame(list = c("sound.files", "selec", "start", "end")) 
1. dataframe <- data.frame('voice.wav', 1, 1, 3)
1. names(dataframe) <- c("sound.files", "selec", "start", "end")
1. write.csv(a, 'test.csv')
