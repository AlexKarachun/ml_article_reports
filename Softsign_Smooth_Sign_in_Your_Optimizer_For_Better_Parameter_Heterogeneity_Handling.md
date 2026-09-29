Softsign: Smooth Sign in Your Optimizer For Better Parameter Heterogeneity Handling
https://openreview.net/forum?id=n47bK7WM3U



поинты
- хотелось бы взять лучшее и от lmo методов и от Steepest descent и получить один хороший метод оптимизации. но через hard switch - это не удачное решение.
- предложили гладкую интерполяцию для lmo в steepest
- показывает прирост метрик. причем прирост появляется именно после начала перехода, когда значительная доля параметров уходит из lmo-style (sign) режима в sgd-like (steepest descent) режим. круто, значит идея преимущества magnitude-aware шагов в окрестности оптимума не лишена смысла.

сильные стороны
- идея простая: одна функция tanh с температурой + расписание температуры. переход параметр-wise, без сброса состояния оптимизатора
- можно не переобучать с нуля: при a_sign = 0.9 берешь чекпоинт Signum/Muon и дообучаешь последние 10%
- по сути один новый гиперпараметр (a_sign), и показана робастность к нему (0.3-0.9) на NCP и 130M
- разнообразные эксперименты: NCP (LSTM, трансформер), GNN на 7 датасетах, LLM 130M/360M/720M, CNN на CIFAR. SoftSignum > Signum и SoftMuon > Muon везде
- есть теория: общая рамка через регуляризатор V и сопряженную по Фенхелю, сходимость в стохастическом невыпуклом случае. tanh получается из энтропийного регуляризатора, а не с потолка

недостатки
- a_sign: дефолт 0.9, и в 5.1 показано, что от 0.3 до 0.9 метрики почти не меняются (NCP и 130M). но для несбалансированных данных такой абляции нет, а там это как раз важно
- не хватило обоснования корректности оценки распределения моментумов только в момент перехода. идея понятная, но не обосновано, что это распределение не сильно поменяется в процессе дальшнейшего обучения.
- В формулировке 2.4 алгебраический soft-sign выглядит взятым с потолка, а теоретическая мотивация появляется только позже. Краткий анонс наличия вывода soft-sign  сделал бы изложение яснее.
- слишком много взято с потолка - линейный шедулинг p_k, folded Cauchy, клип tau >= 1. бейзлайн HardSwitchSign слабый: есть только в NCP и на CIFAR (в LLM и GNN его нет), не описано как подстраивался lr после переключения, и на трансформере он дает ровно столько же сколько Signum (56.17) - переключение там похоже ничего не дало. так что возможно из hardswitch можно выбить метрики и получше. по цифрам преимущество над HardSwitch не минорно (57.05 vs 56.17 при CI +-0.1), а вот над AdamW (57.05 vs 56.91) и на 720M (16.216 vs 16.362) - да
- на сильно не сбаллансированных данных (k >= 100) их обходит обычный Signum. авторы честно пишут, что их зона - средняя несбалансированность, и это ожидаемо: sign-методы сильнее на тяжелых хвостах, а SoftSignum - компромисс. но a_sign = 1 в точности дает Signum, и на CIFAR a_sign тюнится, так что SoftSignum по идее не должен проигрывать Signum вообще. значит мешает что-то еще: клип tau >= 1, аппроксимация Cauchy или тюнинг. 

проверить
- нет ли проблемы, что t->0, a lr=const. ведь на самом деле истинный lr* для малых моментумов это lr* = lr * t, а не lr. ПРОВЕРЕНО: да, авторы это знают - отсюда клип tau >= 1 (ур. 7) и фраза "effective step size is tau*delta" в обсуждении теоремы. но в LLM сверху еще косинусный шедулер lr, так что в хвосте lr режется дважды. плюс теорема доказана для фиксированного tau и не покрывает ни sign-фазу (tau = inf), ни само расписание
- корректно ли сравнение softsignum и softmuon с остальными? ведь в целом у нас появился еще один множетель в норму шага: tau? ПРОВЕРЕНО: сравнение честное. на NCP/GNN/CIFAR все оптимизаторы тюнились Optuna (lr, wd, momentum), а в LLM SoftMuon взял гиперпараметры Muon без дотюна (E.3.1). если что, это в минус авторам


не хватило эксперементов на разных dl задачах/архитектурах/размерах моделей (точнее: набор задач нормальный, но масштаб до 720M, и SoftSignum на 360M/720M не гоняли, только SoftMuon). хочется проверить, тянет ли softsign и softmuon на новый стандарт по умолчанию вместо AdamW и Muon. прям с интересом этим бы занялся. в первую очередь попробовать бы его на разные притрейн llm разных видов и размеров. потом расширить область на остальное потихоньку. покахать, что он хорош, чтобы его заметили и начали использовать в других работах.

еще было бы интересно сделать softadam. Adam формально уже steepest descent, но его нормированный шаг m/sqrt(v) сам по себе почти sign-like - интересно обернуть его в tanh с температурой.

вообщем идея прикольная, но недотестили.






мне кажется было бы здорово вместо сигригации параметров по интерполируемому квантилю, довести все парамерты до sgd честным образом (по конструкции при p -> 1 квантиль Cauchy уходит в бесконечность, tau упирается в клип 1, и к концу почти все координаты с |m| << 1 уже в линейном режиме - но время нормировано на N, и до конца перехода не доходим). да, у нас не нормированное время обучения получается, но было бы интересно посмотреть, что будет, когда все параметры перейдут в зону стабильных градиентов - это будет оптимум или плато. и это будет не обычное sgd-like обучение, ведь мы будем использовать одновременно и sign и sgd like подходы, как авторы и предложили. 


тз доклада: "прокомментируйте основную идею статьи, её сильные стороны и недостатки"


theta_(k+1) = (1 - delta lambda) theta_k - delta tanh(tau_k m_(k+1))

theta_(k+1) = theta_k - delta lambda theta_k - delta tanh(tau_k m_(k+1))

theta_(k+1) = theta_k - delta (lambda theta_k + tanh(tau_k m_(k+1)))


## Softsign: Smooth Sign in Your Optimizer For Better Parameter Heterogeneity Handling

В статье [Softsign: Smooth Sign in Your Optimizer For Better Parameter Heterogeneity Handling](https://openreview.net/forum?id=n47bK7WM3U) авторы предлагают гладкий переход от sign-based оптимизаторов к SGD-like шагам. Методы оптимизации делятся на два класса. Steepest descent (SGD, а в трактовке авторов и Adam) учитывает величину градиента, а LMO-методы (signSGD, Signum, Muon) её отбрасывают и делают шаг фиксированной величины. LMO-методы хорошо работают в начале обучения и устойчивы к тяжелохвостым градиентам, но рядом с оптимумом у них проблема: координаты с почти нулевым градиентом всё равно получают шаг полной величины и осциллируют. SGD-like оптимизаторы эту проблему у оптимума решают, но переключить оптимизатор на SGD в фиксированный момент (hard switch) не очень удачно: все параметры переключаются одновременно, масштабы шагов у Signum и SGD разные, а моментум из sign-фазы не факт что подходит для SGD. Вместо этого авторы заменяют sign(m) на tanh(tau * m). При большом tau это sign, а для координат с малым моментумом это линейный шаг tau * m, по сути momentum SGD с эффективным lr = delta * tau. Температуру снижают по расписанию так, чтобы доля координат в sign-режиме постепенно уменьшалась до 0. Тот же принцип переносится на Muon: sign по сингулярным числам заменяется на гладкую функцию, получается SoftMuon. Так же есть теория: шаг записывается в общем виде через выпуклый регуляризатор V, и для всего семейства доказана сходимость в стохастическом невыпуклом случае.

Мне очень нравится сама идея: одна функция tanh с температурой и расписание. Переход параметр-wise и без сброса состояния оптимизатора, это как раз закрывает проблемы hard switch. Новый гиперпараметр по сути один - alpha_sign, и авторы показывают, что результат к нему малочувствителен: от 0.3 до 0.9 метрики почти не меняются. Теория тоже приятная: в одну рамку попадают SGD, Muon, Lion и SoftSignum, а tanh выводится из регуляризатора, а не берётся с потолка. По экспериментам сетапы довольно разнообразные: LSTM и трансформер на next-character prediction, графовые трансформеры на 7 датасетах, LLM на 130M, 360M и 720M, CNN на CIFAR. Прирост стабильный: SoftSignum лучше Signum, SoftMuon лучше Muon, и SoftMuon в среднем лучший из всех. Здорово, что прирост появляется именно после начала перехода. Значит в окрестности оптимума magnitude-aware шаги действительно нужны.

Первая группа недостатков про то, как метод подан. Много решений без обоснования: линейный шедулинг p_k, клип tau >= 1, оценка распределения моментумов один раз в момент перехода без проверки, что оно дальше не меняется. Алгебраический soft-sign в 2.4 выглядит взятым с потолка, а его вывод появляется только в разделе 3. Краткий анонс наличия обоснования в 2.4 сделал бы изложение яснее.

Вторая группа про убедительность экспериментов. Прирост по абсолютной величине скромный: на 720M perplexity 16.216 против 16.362, на трансформере SoftSignum обходит AdamW всего на 0.14 п.п. Бейзлайн HardSwitchSign слабоват: он есть только в NCP и на CIFAR, и не описано, как подстраивался lr после переключения. Так что тезис "мягкий переход лучше жёсткого" подтверждён слабее, чем хотелось бы. На сильно несбалансированном CIFAR (k >= 100) SoftSignum проигрывает обычному Signum. Авторы пишут, что их зона это средняя несбалансированность, но alpha_sign = 1 в точности даёт Signum, так что SoftSignum по идее вообще не должен проигрывать Signum. Раз проигрывает, мешает что-то ещё: клип tau >= 1, аппроксимация Cauchy или что-то другое. Абляции на alpha_sign для несбалансированных данных нет, а здесь она была бы прям уместна.

В целом идея прикольная: по построению оптимизатор должен быть не слабее своих родителей, и в нём заложен индуктивный биас, который совпадает с нашим пониманием того, как ведут себя градиенты по ходу обучения. Теоретически идея хорошо обоснована, но недотестили. Когда перечисленные выше эксперименты появятся в статье, напрашивается следующая проверка: тянет ли SoftMuon на новый дефолт вместо AdamW и Muon. Я бы для начала сделал претрейн LLM разных размеров и архитектур, а потом попробовал другие области deep learning. Так же было бы интересно сделать SoftAdam: нормированный шаг AdamW m / sqrt(v) сам по себе почти sign-like, и его можно обернуть в tanh с температурой.





## Softsign: Smooth Sign in Your Optimizer For Better Parameter Heterogeneity Handling

In [Softsign: Smooth Sign in Your Optimizer For Better Parameter Heterogeneity Handling](https://openreview.net/forum?id=n47bK7WM3U), the authors propose a smooth transition from sign-based optimizers to SGD-like steps. Optimization methods fall into two classes. Steepest descent (SGD, and in the authors' framing also Adam) takes the gradient magnitude into account, while LMO methods (signSGD, Signum, Muon) drop it and take a step of fixed magnitude. LMO methods work well early in training and are robust to heavy-tailed gradients, but near the optimum they have a problem: coordinates whose gradient is already almost zero still receive a full-magnitude step and oscillate. SGD like optimizers solve this near optimum problem, but switching the optimizer to SGD at a fixed moment (hard switch) is not a great solution: all parameters switch at once, the step scales of Signum and SGD are different, and the momentum accumulated in the sign phase is not necessarily suitable for SGD. Instead, the authors replace sign(m) with tanh(tau * m). For large tau this is sign, and for coordinates with small momentum it is a linear step tau * m, essentially momentum SGD with an effective lr = delta * tau. The temperature is decreased according to a schedule so that the fraction of coordinates in the sign regime gradually goes down to 0. The same principle carries over to Muon: the sign of the singular values is replaced with a smooth function, which gives SoftMuon. There is also theory: the step is written in a general form through a convex regularizer V, and convergence is proved for the whole family in the stochastic non-convex setting.

I really like the core idea: one tanh function with a temperature and a schedule. The transition is parameter-wise and does not reset the optimizer state, which is exactly what fixes the problems of the hard switch. There is essentially one new hyperparameter, alpha_sign, and the authors show that the result is not very sensitive to it: from 0.3 to 0.9 the metrics barely change. The theory is nice too: SGD, Muon, Lion and SoftSignum all fit into one framework, and tanh is derived from a regularizer rather than pulled out of thin air. On the experimental side, the setups are fairly diverse: an LSTM and a Transformer on next-character prediction, graph transformers on 7 datasets, LLMs at 130M, 360M and 720M, a CNN on CIFAR. The gains are consistent: SoftSignum beats Signum, SoftMuon beats Muon, and SoftMuon is the best overall on average. It is great that the gain appears right after the transition starts. So magnitude-aware steps really are needed near the optimum.

The first group of weaknesses is about how the method is presented. Many choices are not justified: the linear schedule for p_k, the clip tau >= 1, and estimating the momentum distribution only once at the transition point without checking that it does not change afterwards. The algebraic soft-sign in Section 2.4 looks pulled out of thin air, and its derivation only appears in Section 3. A brief note in 2.4 that a justification exists would make the presentation clearer.

The second group is about the strength of the evidence. The gains are modest in absolute terms: on the 720M model, perplexity is 16.216 vs 16.362, and on the Transformer SoftSignum beats AdamW by only 0.14 pp. The HardSwitchSign baseline is a bit weak: it only appears in NCP and CIFAR, there is no description of how the lr was adjusted after the switch. So the claim that a smooth transition is better than a hard one is supported less convincingly than I would like. On heavily imbalanced CIFAR (k >= 100), SoftSignum loses to plain Signum. The authors say that their sweet spot is moderate imbalance, but alpha_sign = 1 gives exactly Signum, so in principle SoftSignum should never lose to Signum. Since it does, something else is getting in the way: the clip tau >= 1, the Cauchy approximation, or something else. There is no ablation on alpha_sign for imbalanced data, and it would be really appropriate here.

Overall, the idea is cool: by construction, the optimizer should be at least as strong as its parents, and it carries an inductive bias that matches our understanding of how gradients behave during training. It is well supported theoretically, but undertested. Once the experiments mentioned above are in the paper, the obvious next check is whether SoftMuon can become the new default instead of AdamW and Muon. I would start with LLM pretraining at different sizes and architectures, and then try other areas of deep learning. It would also be interesting to build SoftAdam: the normalized AdamW step m / sqrt(v) is already almost sign-like by itself, and it could be wrapped in tanh with a temperature.
