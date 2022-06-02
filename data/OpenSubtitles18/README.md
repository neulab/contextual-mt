# Downloading OpenSubtitles18 en-fr

First, download and unzip the [OpenSubtitles](https://opus.nlpl.eu/OpenSubtitles-v2018.php) data
```
wget https://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/xml/fr.zip
wget https://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/xml/en.zip
unzip download.php?f=OpenSubtitles%2Fv2018%2Fxml%2Ffr.zip
unzip download.php?f=OpenSubtitles%2Fv2018%2Fxml%2Fen.zip
```

Then, process the data
```
python parser.py --data OpenSubtitles/xml/en/ --output en/
python parser.py --data OpenSubtitles/xml/fr/ --output fr/

```
