Scale-wise Distillation of Diffusion Models
https://openreview.net/forum?id=Z06LNjqU1g




красивая идея. как бы на поверхности и думаешь, блин, как я сам не догадался

поинты
- обосновали что высокочастотные признаки изображения отсутствуют на ранних этапах расшумления и формируются в конце. это значит, что можно не конструировать то, что пока скрыто за сильным шумом.
- дообучили sota DM расшумлять в разных latent resolution

сильные стороны
- кратно ускоряет генерацию без потери качества
- подходит для генерации и картинок и видео
- встраивается в большой класс sota моделей
- хорошо, чисто описана и обоснована идея.
- еще и метрики качества стабильно повышает

недостатки
- не понятно, как собрать достаточно разнообразный (по модам) датасет для дообучения 
- модели надо дообучать на это. пусть это и не критично, но дополнительных усилий требует - не plug and play
- метод подходит для класса неиросетей диффузионок допускающего любые входные разрешения (strong transorfer align). да, сейчас это sota в генерации high quality diverse картинок, но не факт, что это будет актуально в будующем или других областях где применяют диффузионки.
- Мне не хватает проверки сохранения разнообразия генераций после дистилляции. Рост метрик предпочтения сам по себе не показывает, насколько хорошо студент сохраняет редкие варианты из распределения учителя.

было бы интересно попробовать для текстовой диффузии, ведь там тоже есть и общий посыл текста, и посыл абзацев и посыл предложений, а не только отдельных токенов



тз доклада: "прокомментируйте основную идею статьи, её сильные стороны и недостатки"





## Scale-wise Distillation of Diffusion Models







## Scale-wise Distillation of Diffusion Models

В статье [Scale-wise Distillation of Diffusion Models](https://openreview.net/forum?id=Z06LNjqU1g) авторы предлагают ускорять генерацию диффузионных моделей за счёт постепенного увеличения разрешения по ходу расшумления. Спектральный анализ VAE-латентов показыал, что на ранних шагах высокочастотные детали в основном скрыты шумом и проявляются только ближе к концу и поэтому на ранних шагах можно не считать всё в полном разрешении, ведь мы все равно еще не восстановить эти мелкие детали. Авторы дистиллируют предобученную модель так, чтобы она начинала с латента маленького размера и увеличивала его на каждом шаге. Для видео то же самое делается и по пространственному, и по временному разрешению. Так же они предлагают MMD-loss, который сопоставляет признаки сгенерированных и целевых примеров, извлечённые моделью-учителем. 

Больше всего мне понравилась сама идея. Она вроде бы на поверхности - словом красивая. При этом авторы её хорошо обосновывают и проверяют экспериментально. Большой плюс, что генерацию с растущим и фиксированным разрешением сравнивают и при одинаковом числе шагов, и при сопоставимом времени генерации - чистая работа, приятно. В первом случае метод даёт примерно двукратное ускорение для изображений и трёхкратное для видео без заметной потери качества по проведённым оценкам, во втором за то же время получается более качественный результат. Метод проверен на нескольких семействах моделей (SDXL, SD3.5, FLUX, Wan2.1), то есть встраивается в большой класс современных моделей и для картинок, и для видео. Удивило, что выигрыш не только в скорости и многие метрики качества тоже растут, и это подтверждается оценкой людей. Правда, не по всем критериям сразу: местами проседает FID и встречаются отдельные потери по дефектам.

Но есть и недостатки, например модель надо дообучать - не plug and play. Модель нужно адаптировать к переходам между масштабами, а для этого собрать данные и запустить дистилляцию. Хотя в целом это не критично. Так же метод предполагает, что модель умеет работать на разных разрешениях. Ещё мне не хватило обоснования сохранения разнообразия генераций при дистилляции. Студент обучается на синтетических данных учителя, и не понятно, насколько такой датасет покрывает редкие моды его распределения. Рост метрик предпочтения сам по себе этого не показывает, тк средняя картинка может стать лучше, а часть вариантов при этом потеряться. Надо бы проверить, как результат зависит от объёма датасета и разнообразия обучающих промптов, и сравнить вариативность генераций учителя и студента на одинаковых запросах. Это не претензия к самому методу, а скорее к полноте проверки.

Так же было бы интересно попробовать похожую идею в текстовой диффузии. В тексте тоже есть разные уровни структуры: общий замысел, содержание абзацев, предложений и отдельных токенов. Эти уровни нельзя напрямую приравнять к разрешению изображения, так что сначала пришлось бы определить, что считать масштабом в такой модели. Но идейно метод можно попробовать переложить на текстовую диффузию и я прям не сомневаюсь в том, что его можно там завести.




## Scale-wise Distillation of Diffusion Models

In [Scale-wise Distillation of Diffusion Models](https://openreview.net/forum?id=Z06LNjqU1g), the authors propose speeding up diffusion models by gradually increasing the resolution during denoising. Spectral analysis of VAE latents showed that at early steps the high-frequency details are mostly hidden by noise and only appear closer to the end, so there is no need to compute everything at full resolution early on, since we cannot recover these fine details yet anyway. The authors distill a pretrained model so that it starts from a small latent and increases its size at every step. For video, the same is done for both spatial and temporal resolution. They also propose an MMD loss that matches features of generated and target samples extracted by the teacher model.

What I liked most is the idea itself. It seems to lie on the surface, in short, it is just elegant. At the same time, the authors justify it well and verify it experimentally. A big plus is that they compare generation with increasing and fixed resolution both at the same number of steps and at comparable generation time. Clean work, nice to see. In the first case, the method gives roughly a 2x speedup for images and 3x for video with no noticeable loss in quality according to the reported evaluations, and in the second case it produces a better result in the same amount of time. The method is tested on several model families (SDXL, SD3.5, FLUX, Wan2.1), so it fits into a large class of modern models for both images and video. I was surprised that the gain is not only in speed: many quality metrics also improve, and this is confirmed by human evaluation. Although not on every criterion at once: FID drops in some cases, and there are a few losses in terms of defects.

But there are also weaknesses. For example, the model has to be fine-tuned, so it is not plug and play. The model needs to be adapted to the transitions between scales, which means collecting data and running the distillation. Overall, though, this is not critical. The method also assumes that the model can work at different resolutions. I also felt that the paper lacks evidence that diversity is preserved during distillation. The student is trained on synthetic data from the teacher, and it is unclear how well such a dataset covers the rare modes of its distribution. Growth in preference metrics does not show this by itself, since the average image may get better while some variants get lost. It would be worth checking how the result depends on the dataset size and the diversity of training prompts, and comparing the variability of teacher and student generations on the same prompts. This is not a criticism of the method itself, but rather of the completeness of the evaluation.

It would also be interesting to try a similar idea in text diffusion. Text also has different levels of structure: the overall message, the content of paragraphs, sentences, and individual tokens. These levels cannot be directly mapped to image resolution, so one would first have to define what counts as scale in such a model. But conceptually the method could be carried over to text diffusion, and I have no doubt that it can be made to work there.
