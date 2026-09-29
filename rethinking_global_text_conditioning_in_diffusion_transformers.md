# Rethinking Global Text Conditioning in Diffusion Transformers
https://openreview.net/forum?id=65Ai8mLfjI

в статье проводят абляционное исследование влияния промпт составляющей вектора модуляции на различных диффузионных моделях (FLUX schell, HiDream-Fast, COSMOS) для различных задач: text to image, image editing.


сильные стороны
- прирост метрик стабильный
- особенно интересно и неожиданно оказалось, что такой подход позволяет выполнить требования по количеству объектов и корректности рук
- прикольно что нашли что модуляция вносит малый вклад в качество - прям неожиданно. вообще делать проверки рудиментарность используемых техник - теперь кажется интересной нишей
- практически не увеличивает компьют и training free переносится на большой класс современных моделей



недостатки
- улучшение качества генерации наблюдается при w=3 (>1). это просто в среднем может скейлить вектор модуляции. надо проверить, не в этом ли дело. Как я бы сделал такую проверку: собрать метрики по картикам сгенерированным четырьмя пайплайнами:
 1. vanilla: модуляция с обычным вектором (y(p, t))
 2. scaled: модуляция с обычным заскейлиным вектором (y(p, t) * alpha)
 2. guided: модуляция с guided вектором (y(p, p+, p-, t))
 2. guided: модуляция с guided заскейлиным вектором (y(p, p+, p-, t) / alpha)
и надо посмотреть, что конкретно улучшает параметры: гайдес или скейлинг вектора модуляции.
- я вижу ценность в том что бы доделать пайплайн до end to end алгоритма генерации, для этого надо автоматически генерировать два дополнительных промпта через llm. было бы интересно провести анализ разных llm с целлью найти самый дешевый по compute/vram/api cost и все еще выдающий хороший результат. 
- за счет чего clip передает количество объектов - это неожиданно. хотелось бы исследовать это место. и есть статья про то, что клип плохо переносит цифры. https://arxiv.org/html/2409.15035





тз доклада: "прокомментируйте основную идею статьи, её сильные стороны и недостатки"


## Rethinking Global Text Conditioning in Diffusion Transformers

В статье [Rethinking Global Text Conditioning in Diffusion Transformers](https://openreview.net/forum?id=65Ai8mLfjI) авторы разбираются, насколько диффузионным трансформерам нужен текстовый сигнал в векторе модуляции. Как правило, информация из промпта поступает в модель через joint attention и модуляцию CLIP эмбеддинга. Авторы показывают, что второй способ часто почти не влияет на результат: в HiDream-Fast и FLUX эмбеддинг CLIP можно занулить практически без потери качества. Добавление такого обуславливания в COSMOS тоже само по себе не улучшает метрики. Но авторы так же предлагают способ сделать этот сигнал полезным. Они предлагают modulation guidance: к исходному вектору модуляции добавляется взвешенная разность векторов для двух дополнительных промптов: описывающих желательные и нежелательные свойства изображения. Силу guidance можно менять между слоями, чтобы улучшать качество и при этом лучше сохранять соответствие исходному промпту.

Больше всего мне понравилось само абляционное исследование. Кажется логичным, что текстовая составляющая модуляции должна помогать модели, но эксперименты показывают, что в обычно она часто почти ничего не даёт. После этой статьи  стало интересно, какие ещё привычне методы современных моделей являются рудиментами. Так же из плюсов предложенного метода генерации: он почти не увеличивает compute, работает на разных моделях в задачах генерации изображений, видео и редактирования и стабильно улучшает метрики. Особенно неожиданно, что такой простой подход помогает точнее соблюдать количество объектов и уменьшать ошибки при генерации рук. Для моделей, где текстовое обуславливание через модуляцию уже есть, дообучение не требуется, а для моделей без него сначала нужно обучить небольшой MLP - тоже здорово.

При этом мне не хватило проверки, за счёт чего именно получается прирост: изменения направления вектора модуляции или его нормы. Авторы часто используют w = 3, и это вызывает у меня настороженность, что часть улучшения может объясняться просто масштабированием вектора модуляции. Само значение w > 1 этого, конечно, не доказывает, надо сравнить несколько пайплайнов генерации: с guidance (y_guide = y(p, p+, p-, t)) и без него (y_base = y(p, t)), а так же варианты с симметричными нормами: с guidance (y_guide * norm(y_base) / norm(y_guide)) и без него (y_base norm(y_guide) / norm(y_base)). Получатся четыре варианта, которые позволят проверить, что дает прирост: увеличенная норма или гайденс.

Так же для выпуска в продакшн метод требует доработки. Сейчас дополнительные промпты задаются человеком под конкретные задачи. Можно попробовать генерировать их через LLM на основе пользовательского запроса. Тогда стоило бы сравнить разные LLM и найти наиболее дешёвую по вычислениям и памяти или стоимости API, которая всё ещё даёт хорошие результаты. Это не камень в огорд авторов, а скорее возможное продолжение работы - для удобного применения метода такая проверка была бы полезна.

Отдельный вопрос у меня вызвал прирост качества генерации заданного количества объектов. В статье есть анализ того, как guidance меняет attention при исправлении рук, а для счёта хотелось бы увидеть похожее исследование. В работе «[Can CLIP Count Stars?](https://arxiv.org/html/2409.15035)» показано, что CLIP плохо справляется с представлением количества, поэтому результат авторов рассматриваемой статьи особенно неожиданен. Это не значит, что статьи противоречат друг другу: CLIP может давать полезный сигнал диффузионной модели, даже если сам не умеет надёжно считать объекты. Но хотелось бы лучше понять, как этот сигнал помогает модели и какую роль здесь играет взаимодействие с текстовыми токенами в attention.




## Rethinking Global Text Conditioning in Diffusion Transformers

In [Rethinking Global Text Conditioning in Diffusion Transformers](https://openreview.net/forum?id=65Ai8mLfjI), the authors investigate how much diffusion transformers need the text signal in the modulation vector. Prompt information usually reaches the model through joint attention and modulation based on a CLIP embedding. The authors show that the latter often has almost no effect on the result: in HiDream-Fast and FLUX, the CLIP embedding can be set to zero with almost no loss in quality. Adding this kind of conditioning to COSMOS does not improve the metrics on its own either. However, the authors also suggest a way to make this signal useful. They propose modulation guidance: adding a weighted difference between the vectors for two additional prompts, describing desirable and undesirable image properties, to the original modulation vector. The guidance strength can vary across layers to improve quality while better preserving alignment with the original prompt.

What I liked most was the ablation study itself. It seems reasonable that the text component of modulation should help the model, but the experiments show that in its usual form it often contributes very little. The paper made me wonder which other common techniques in modern models might be redundant. The proposed generation method also has practical strengths: it adds almost no compute, works across different models for image generation, video generation, and image editing, and consistently improves the metrics. I found it especially surprising that such a simple approach helps generate the correct number of objects and reduce errors in generated hands. Models that already have modulation-based text conditioning do not need fine-tuning, while models without it only require training a small MLP, which is also a useful advantage.

Still, I felt that one experiment was missing: checking whether the gains come from changing the direction of the modulation vector or its norm. The authors often use w = 3, which makes me suspect that part of the improvement might come simply from scaling the modulation vector. Of course, w > 1 alone does not prove this. I would compare several generation pipelines: with guidance (y_guide = y(p, p+, p-, t)), without it (y_base = y(p, t)), and two versions with swapped norms: guided with the baseline norm (y_guide * norm(y_base) / norm(y_guide)) and unguided with the guided norm (y_base * norm(y_guide) / norm(y_base)). These four variants would help test whether the gains come from the increased norm or from guidance.

The method also needs some further work before it can be used in production. Currently, the additional prompts are written by a person for specific tasks. One option would be to generate them with an LLM based on the user's request. It would then be worth comparing different LLMs to find the cheapest one in terms of compute and memory usage or API cost that still produces good results. I see this as a possible extension of the work rather than a criticism of the authors, but such an experiment would be useful for making the method easier to apply.

Another question I had concerns the improvement in generating a specified number of objects. The paper analyzes how guidance changes attention when correcting hands, and I would like to see a similar analysis for counting. [Can CLIP Count Stars?](https://arxiv.org/html/2409.15035) shows that CLIP struggles to represent quantity, which makes this result especially surprising. This does not mean the two papers contradict each other: CLIP can provide a useful signal to a diffusion model even if it cannot reliably count objects itself. Still, I would like to understand better how this signal helps the model and what role its interaction with text tokens in attention plays.
