# Diplomová práca

Cieľom práce je detegovať vaskulárnu leukoencefalopatiu v CT snímkach mozgu s využitím neurónových sietí.

Vaskulárna leukoencefalopatia je spôsobená starnutim mozgu a opakovanými drobnými porážkami mozgu na podklade uzáveru drobných ciev, väčšinou bez prejavu mozgovej porážky. Na CT snímkach sa zobrazí ako tmavšie zóny v blízkosti bočných komôr (uložených centrálne - ktoré sú pre obsah vody takmer čierne).

CT snímky sú uložené vo formáte DICOM (.dcm), ktoré treba zjednotiť v rozmeroch a spracovať. Výsledné očistené dáta sú použité na trénovanie 3D konvolučnej neurónovej siete.

## Knižnice python

Použité knižnice v projekte:
- os
- pydicom
- numpy
- scipy
- matplotlib
- skimage
- cv2
