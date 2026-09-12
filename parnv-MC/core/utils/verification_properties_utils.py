import ast
import json
from typing import Dict

from core.configuration import consts

TEST_PROPERTY_ACAS = {

    # the following properties are properties 1,2,3,4 from MarabouApplications
    "property_1": {
        "type": "basic",
        "input":
            [
                (0, {"Lower": 0.6, "Upper": 0.6798577687}),
                (1, {"Lower": -0.5, "Upper": 0.5}),
                (2, {"Lower": -0.5, "Upper": 0.5}),
                (3, {"Lower": 0.45, "Upper": 0.5}),
                (4, {"Lower": -0.5, "Upper": -0.45}),
            ],
        "output":
            [
                (0, {"Lower": 3.9911256459}),
            ]
    },
    "test_mini": {
        "type": "basic",
        "input":
            [
                (0, {"Lower": 0.5, "Upper": 1}),
                (1, {"Lower": 1, "Upper": 2}),
            ],
        "output":
            [
                # representation of inequasion: +1*y0 -1*y1 <= 0
                ([(1, 0), (-1, 1)], {"Upper": 0}),
                # representation of inequasion: +1*y0 -1*y2 <= 0
                ([(1, 0), (-1, 2)], {"Upper": 0}),
                # representation of inequasion: +1*y0 -1*y3 <= 0
                ([(1, 0), (-1, 3)], {"Upper": 0}),
                # representation of inequasion: +1*y0 -1*y4 <= 0
                ([(1, 0), (-1, 4)], {"Upper": 0}),
            ]
    },
    "property_2": {
        "type": "acas_xu_conjunction",
        "input":
            [
                (0, {"Lower": 0.6, "Upper": 0.6798577687}),
                (1, {"Lower": -0.5, "Upper": 0.5}),
                (2, {"Lower": -0.5, "Upper": 0.5}),
                (3, {"Lower": 0.45, "Upper": 0.5}),
                (4, {"Lower": -0.5, "Upper": -0.45}),
            ],
        "output":
            [
                # representation of inequasion: +1*y0 -1*y1 <= 0
                ([(1, 0), (-1, 1)], {"Upper": 0}),
                # representation of inequasion: +1*y0 -1*y2 <= 0
                ([(1, 0), (-1, 2)], {"Upper": 0}),
                # representation of inequasion: +1*y0 -1*y3 <= 0
                ([(1, 0), (-1, 3)], {"Upper": 0}),
                # representation of inequasion: +1*y0 -1*y4 <= 0
                ([(1, 0), (-1, 4)], {"Upper": 0}),
            ]
    },
    "property_3": {
        "type": "acas_xu_conjunction",
        "input":
            [
                (0, {"Lower": -0.21751900893722352, "Upper": -0.21751900893722352}),
                (1, {"Lower": -0.2669214890001048, "Upper": -0.2669214890001048}),
                (2, {"Lower": -0.41246122375995143, "Upper": -0.41246122375995143}),
                (3, {"Lower": -0.16000000137052063, "Upper": -0.16000000137052063}),
                (4, {"Lower": -0.02502788748005025, "Upper": -0.02502788748005025}),
            ],
        "output":
            [
                # representation of inequasion: +1*y0 -1*y1 <= 0
                ([(1, 0), (-1, 1)], {"Upper": 0}),
                # representation of inequasion: +1*y0 -1*y2 <= 0
                ([(1, 0), (-1, 2)], {"Upper": 0}),
                # representation of inequasion: +1*y0 -1*y3 <= 0
                ([(1, 0), (-1, 3)], {"Upper": 0}),
                # representation of inequasion: +1*y0 -1*y4 <= 0
                ([(1, 0), (-1, 4)], {"Upper": 0}),
            ]
    },
    "property_4": {
        "type": "acas_xu_conjunction",
        "input":
            [
                (0, {"Lower": -0.22751900893722352, "Upper": -0.2075190089372235}),
                (1, {"Lower": -0.2769214890001048, "Upper": -0.2569214890001048}),
                (2, {"Lower": -0.42246122375995143, "Upper": -0.4024612237599514}),
                (3, {"Lower": -0.17000000137052063, "Upper": -0.1500000013705206}),
                (4, {"Lower": -0.03502788748005025, "Upper": -0.015027887480050248}),
            ],
        "output":
            [
                # representation of inequasion: +1*y0 -1*y1 <= 0
                ([(1, 0), (-1, 1)], {"Upper": 0}),
                # representation of inequasion: +1*y0 -1*y2 <= 0
                ([(1, 0), (-1, 2)], {"Upper": 0}),
                # representation of inequasion: +1*y0 -1*y3 <= 0
                ([(1, 0), (-1, 3)], {"Upper": 0}),
                # representation of inequasion: +1*y0 -1*y4 <= 0
                ([(1, 0), (-1, 4)], {"Upper": 0}),
            ]
    },

    # the following properties are used for sanity
    "sanity_UNSAT": {
        "type": "basic",
        "input":
            [
                (0, {"Lower": 0.6, "Upper": 0.6798577687}),
                (1, {"Lower": -0.5, "Upper": 0.5}),
                (2, {"Lower": -0.5, "Upper": 0.5}),
                (3, {"Lower": 0.45, "Upper": 0.5}),
                (4, {"Lower": -0.5, "Upper": -0.45}),
            ],
        "output":
            [
                (0, {"Lower": 1000000000000000000000000000000000000000000000.0}),
            ]
    },
    "sanity_SAT": {
        "type": "basic",
        "input":
            [
                # (0, {"Lower": 0.6, "Upper": 0.6}),
                # (1, {"Lower": -0.5, "Upper": -0.5}),
                # (2, {"Lower": -0.5, "Upper": -0.5}),
                # (3, {"Lower": 0.45, "Upper": 0.45}),
                # (4, {"Lower": -0.45, "Upper": -0.45}),
                (0, {"Lower": 0.6, "Upper": 0.6798577687}),
                (1, {"Lower": -0.5, "Upper": 0.5}),
                (2, {"Lower": -0.5, "Upper": 0.5}),
                (3, {"Lower": 0.45, "Upper": 0.5}),
                (4, {"Lower": -0.5, "Upper": -0.45}),
            ],
        "output":
            [
                (0, {"Lower": -988888888888888.0}),
            ]
    },
    # the following properties are used for the experiments of CAV_2020 Fig8
    # actually, same as property 1 with bigger output lower bound
    "basic_0": {
        "type": "basic",
        "input":
            [
                (0, {"Lower": 0.6, "Upper": 0.6798577687}),
                (1, {"Lower": -0.5, "Upper": 0.5}),
                (2, {"Lower": -0.5, "Upper": 0.5}),
                (3, {"Lower": 0.45, "Upper": 0.5}),
                (4, {"Lower": -0.5, "Upper": -0.45}),
            ],
        "output":
            [
                (0, {"Lower": 3.9911256459}),
            ]
    },
    "basic_1": {
        "type": "basic",
        "input":
            [
                (0, {"Lower": 0.6, "Upper": 0.6798577687}),
                (1, {"Lower": -0.5, "Upper": 0.5}),
                (2, {"Lower": -0.5, "Upper": 0.5}),
                (3, {"Lower": 0.45, "Upper": 0.5}),
                (4, {"Lower": -0.5, "Upper": -0.45}),
            ],
        "output":
            [
                (0, {"Lower": 100.0}),
            ]
    },
    "basic_2": {
        "type": "basic",
        "input":
            [
                (0, {"Lower": 0.6, "Upper": 0.6798577687}),
                (1, {"Lower": -0.5, "Upper": 0.5}),
                (2, {"Lower": -0.5, "Upper": 0.5}),
                (3, {"Lower": 0.45, "Upper": 0.5}),
                (4, {"Lower": -0.5, "Upper": -0.45}),
            ],
        "output":
            [
                (0, {"Lower": 500.0}),
            ]
    },
    "basic_3": {
        "type": "basic",
        "input":
            [
                (0, {"Lower": 0.6, "Upper": 0.6798577687}),
                (1, {"Lower": -0.5, "Upper": 0.5}),
                (2, {"Lower": -0.5, "Upper": 0.5}),
                (3, {"Lower": 0.45, "Upper": 0.5}),
                (4, {"Lower": -0.5, "Upper": -0.45}),
            ],
        "output":
            [
                (0, {"Lower": 750.0}),
            ]
    },
    "basic_4": {
        "type": "basic",
        "input":
            [
                (0, {"Lower": 0.6, "Upper": 0.6798577687}),
                (1, {"Lower": -0.5, "Upper": 0.5}),
                (2, {"Lower": -0.5, "Upper": 0.5}),
                (3, {"Lower": 0.45, "Upper": 0.5}),
                (4, {"Lower": -0.5, "Upper": -0.45}),
            ],
        "output":
            [
                (0, {"Lower": 1000.0}),
            ]
    },
    # the following properties are used for the experiments of CAV_2020 Fig9
    # notice: most of the outputs are equal, but the inputs are different
    "adversarial_0": {
        "type": "adversarial",
        "input":
            [
                (0, {"Lower": 0.19920080677521143, "Upper": 0.2792008067752114}),
                (1, {"Lower": -0.017208202294050033, "Upper": 0.06279179770594998}),
                (2, {"Lower": -0.10919992573920609, "Upper": -0.02919992573920608}),
                (3, {"Lower": -0.23510547902474485, "Upper": -0.15510547902474484}),
                (4, {"Lower": -0.12787478602840877, "Upper": -0.04787478602840876}),
            ],
        "output":
            [
                (0, {"Lower": -0.020547071432855293, "Upper": -0.020547071432855293}),
                (1, {"Lower": 0.019108227270628716, "Upper": 0.019108227270628716}),
                (2, {"Lower": -0.019247074474943282, "Upper": -0.019247074474943282}),
                (3, {"Lower": 0.01906232886097115, "Upper": 0.01906232886097115}),
                (4, {"Lower": -0.01658198836248747, "Upper": -0.01658198836248747}),
            ]
    },

    # notice: most (sometimes all) outputs are equal, but inputs are different
    "0": {
        "input":
            [
                (0, {"Lower": -0.2511004611964829, "Upper": -0.051100461196482844}),
                (1, {"Lower": -0.15252914190674147, "Upper": 0.047470858093258544}),
                (2, {"Lower": -0.17938744999372505, "Upper": 0.02061255000627496}),
                (3, {"Lower": -0.033943630638619965, "Upper": 0.16605636936138005}),
                (4, {"Lower": -0.09258711301318504, "Upper": 0.10741288698681498}),
            ],
        "output":
            [
                (0, {"Lower": -0.1201359, "Upper": 0.07986410000000001}),
                (1, {"Lower": -0.1193657, "Upper": 0.0806343}),
                (2, {"Lower": -0.11944500000000001, "Upper": 0.080555}),
                (3, {"Lower": -0.11923740000000001, "Upper": 0.0807626}),
                (4, {"Lower": -0.119343, "Upper": 0.080657}),
            ]
    },
    "1": {
        "input":
            [
                # (0, {"Lower": 0.5613092934866438, "Upper": 0.5613092934866438}),
                # (1, {"Lower": -0.25473408621509763, "Upper": -0.25473408621509763}),
                # (2, {"Lower": 0.39849547921922923, "Upper": 0.39849547921922923}),
                # (3, {"Lower": -0.3145693757727118, "Upper": -0.3145693757727118}),
                # (4, {"Lower": -0.30928556990994716, "Upper": -0.30928556990994716}),
                (0, {"Lower": 0.5613092934866438, "Upper": 0.7613092934866438}),
                (1, {"Lower": -0.25473408621509763, "Upper": -0.054734086215097594}),
                (2, {"Lower": 0.39849547921922923, "Upper": 0.5984954792192292}),
                (3, {"Lower": -0.3145693757727118, "Upper": -0.11456937577271184}),
                (4, {"Lower": -0.30928556990994716, "Upper": -0.10928556990994712}),
            ],
        "output":
            [
                (0, {"Lower": -0.1201359, "Upper": 0.07986410000000001}),
                (1, {"Lower": -0.1193657, "Upper": 0.0806343}),
                (2, {"Lower": -0.11944500000000001, "Upper": 0.080555}),
                (3, {"Lower": -0.11923740000000001, "Upper": 0.0807626}),
                (4, {"Lower": -0.119343, "Upper": 0.080657}),
            ]
    },
    "2": {
        "input":
            [
                # (0, {"Lower": -0.18475016743442682, "Upper": -0.18475016743442682}),
                # (1, {"Lower": 0.33081368359346197, "Upper": 0.33081368359346197}),
                # (2, {"Lower": 0.39406534577882335, "Upper": 0.39406534577882335}),
                # (3, {"Lower": -0.43224469513207686, "Upper": -0.43224469513207686}),
                # (4, {"Lower": -0.09554150357464855, "Upper": -0.09554150357464855}),
                (0, {"Lower": -0.18475016743442682, "Upper": 0.01524983256557319}),
                (1, {"Lower": 0.33081368359346197, "Upper": 0.5308136835934619}),
                (2, {"Lower": 0.39406534577882335, "Upper": 0.5940653457788233}),
                (3, {"Lower": -0.43224469513207686, "Upper": -0.23224469513207688}),
                (4, {"Lower": -0.09554150357464855, "Upper": 0.10445849642535146}),
            ],
        "output":
            [
                (0, {"Lower": -0.1201359, "Upper": 0.07986410000000001}),
                (1, {"Lower": -0.1193657, "Upper": 0.0806343}),
                (2, {"Lower": -0.11944500000000001, "Upper": 0.080555}),
                (3, {"Lower": -0.11923740000000001, "Upper": 0.0807626}),
                (4, {"Lower": -0.119343, "Upper": 0.080657}),
            ]
    },
    "3": {
        "type": "adversarial",
        "input":
            [
                (0, {"Lower": -0.09843991404369865, "Upper": -0.09843991404369865}),
                (1, {"Lower": -0.006586213254008461, "Upper": -0.006586213254008461}),
                (2, {"Lower": 0.3116869634113456, "Upper": 0.3116869634113456}),
                (3, {"Lower": 0.2877924070846184, "Upper": 0.2877924070846184}),
                (4, {"Lower": 0.3083532209006685, "Upper": 0.3083532209006685}),
                # (0, {"Lower": -0.09843991404369865, "Upper": 0.10156008595630137}),
                # (1, {"Lower": -0.006586213254008461, "Upper": 0.19341378674599155}),
                # (2, {"Lower": 0.3116869634113456, "Upper": 0.5116869634113456}),
                # (3, {"Lower": 0.2877924070846184, "Upper": 0.48779240708461835}),
                # (4, {"Lower": 0.3083532209006685, "Upper": 0.5083532209006685}),
                # (0, {"Lower": -10.0, "Upper": 10.0}),
                # (1, {"Lower": -10.0, "Upper": 10.0}),
                # (2, {"Lower": -10.0, "Upper": 10.0}),
                # (3, {"Lower": -10.0, "Upper": 10.0}),
                # (4, {"Lower": -10.0, "Upper": 10.0}),
            ],
        "output":
            [
                (0, {"Lower": -0.1201359, "Upper": 0.07986410000000001}),
                (1, {"Lower": -0.1193657, "Upper": 0.0806343}),
                (2, {"Lower": -0.11944500000000001, "Upper": 0.080555}),
                (3, {"Lower": -0.11923740000000001, "Upper": 0.0807626}),
                (4, {"Lower": -0.119343, "Upper": 0.080657}),
            ]
    },
    "4": {
        "input":
            [
                # (0, {"Lower": 0.12088057030212532, "Upper": 0.12088057030212532}),
                # (1, {"Lower": -0.29944037519743616, "Upper": -0.29944037519743616}),
                # (2, {"Lower": -0.2131922610670343, "Upper": -0.2131922610670343}),
                # (3, {"Lower": -0.579777264559382, "Upper": -0.579777264559382}),
                # (4, {"Lower": 0.22219188979996382, "Upper": 0.22219188979996382}),
                (0, {"Lower": 0.12088057030212532, "Upper": 0.32088057030212536}),
                (1, {"Lower": -0.29944037519743616, "Upper": -0.09944037519743618}),
                (2, {"Lower": -0.2131922610670343, "Upper": -0.013192261067034278}),
                (3, {"Lower": -0.579777264559382, "Upper": -0.379777264559382}),
                (4, {"Lower": 0.22219188979996382, "Upper": 0.4221918897999638}),
            ],
        "output":
            [
                (0, {"Lower": -0.1201359, "Upper": 0.07986410000000001}),
                (1, {"Lower": -0.1193657, "Upper": 0.0806343}),
                (2, {"Lower": -0.11944500000000001, "Upper": 0.080555}),
                (3, {"Lower": -0.11923740000000001, "Upper": 0.0807626}),
                (4, {"Lower": -0.119343, "Upper": 0.080657}),
            ]
    },
    "5": {
        "input":
            [
                # (0, {"Lower": 0.5510566179903064, "Upper": 0.5510566179903064}),
                # (1, {"Lower": -0.39704931745399785, "Upper": -0.39704931745399785}),
                # (2, {"Lower": -0.42485207458190044, "Upper": -0.42485207458190044}),
                # (3, {"Lower": -0.0882241353439496, "Upper": -0.0882241353439496}),
                # (4, {"Lower": 0.01087717304563493, "Upper": 0.01087717304563493}),
                (0, {"Lower": 0.5510566179903064, "Upper": 0.7510566179903063}),
                (1, {"Lower": -0.39704931745399785, "Upper": -0.19704931745399787}),
                (2, {"Lower": -0.42485207458190044, "Upper": -0.22485207458190046}),
                (3, {"Lower": -0.0882241353439496, "Upper": 0.11177586465605041}),
                (4, {"Lower": 0.01087717304563493, "Upper": 0.21087717304563494}),
            ],
        "output":
            [
                (0, {"Lower": -0.1201359, "Upper": 0.07986410000000001}),
                (1, {"Lower": -0.1193657, "Upper": 0.0806343}),
                (2, {"Lower": -0.11944500000000001, "Upper": 0.080555}),
                (3, {"Lower": -0.11923740000000001, "Upper": 0.0807626}),
                (4, {"Lower": -0.119343, "Upper": 0.080657}),
            ]
    },
    "6": {
        "input":
            [
                # (0, {"Lower": -0.095952965491269, "Upper": -0.095952965491269}),
                # (1, {"Lower": 0.20381010772715577, "Upper": 0.20381010772715577}),
                # (2, {"Lower": -0.43100243118807446, "Upper": -0.43100243118807446}),
                # (3, {"Lower": -0.23890072055215292, "Upper": -0.23890072055215292}),
                # (4, {"Lower": 0.29896458464063347, "Upper": 0.29896458464063347}),
                (0, {"Lower": -0.095952965491269, "Upper": 0.10404703450873101}),
                (1, {"Lower": 0.20381010772715577, "Upper": 0.40381010772715575}),
                (2, {"Lower": -0.43100243118807446, "Upper": -0.23100243118807443}),
                (3, {"Lower": -0.23890072055215292, "Upper": -0.038900720552152906}),
                (4, {"Lower": 0.29896458464063347, "Upper": 0.4989645846406334}),
            ],
        "output":
            [
                (0, {"Lower": -0.1201359, "Upper": 0.07986410000000001}),
                (1, {"Lower": -0.1193657, "Upper": 0.0806343}),
                (2, {"Lower": -0.11944500000000001, "Upper": 0.080555}),
                (3, {"Lower": -0.11923740000000001, "Upper": 0.0807626}),
                (4, {"Lower": -0.119343, "Upper": 0.080657}),
            ]
    },
    "7": {
        "input":
            [
                # (0, {"Lower": 0.5590162013101513, "Upper": 0.5590162013101513}),
                # (1, {"Lower": 0.18933955034545077, "Upper": 0.18933955034545077}),
                # (2, {"Lower": -0.34267058609873347, "Upper": -0.34267058609873347}),
                # (3, {"Lower": -0.033145103880183574, "Upper": -0.033145103880183574}),
                # (4, {"Lower": -0.3155532407300927, "Upper": -0.3155532407300927}),
                (0, {"Lower": 0.5590162013101513, "Upper": 0.7590162013101512}),
                (1, {"Lower": 0.18933955034545077, "Upper": 0.38933955034545076}),
                (2, {"Lower": -0.34267058609873347, "Upper": -0.14267058609873343}),
                (3, {"Lower": -0.033145103880183574, "Upper": 0.16685489611981644}),
                (4, {"Lower": -0.3155532407300927, "Upper": -0.11555324073009268}),
            ],
        "output":
            [
                (0, {"Lower": -0.1201359, "Upper": 0.07986410000000001}),
                (1, {"Lower": -0.1193657, "Upper": 0.0806343}),
                (2, {"Lower": -0.11944500000000001, "Upper": 0.080555}),
                (3, {"Lower": -0.11923740000000001, "Upper": 0.0807626}),
                (4, {"Lower": -0.119343, "Upper": 0.080657}),
            ]
    },
    "8": {
        "input":
            [
                (0, {"Lower": 0.3649481156241908, "Upper": 0.5649481156241909}),
                (1, {"Lower": 0.3815255191938711, "Upper": 0.5815255191938711}),
                (2, {"Lower": -0.09036414812907193, "Upper": 0.10963585187092809}),
                (3, {"Lower": -0.4701356950632751, "Upper": -0.27013569506327517}),
                (4, {"Lower": -0.5615225612619623, "Upper": -0.36152256126196225}),
            ],
        "output":
            [
                (0, {"Lower": -0.1201359, "Upper": 0.07986410000000001}),
                (1, {"Lower": -0.1193657, "Upper": 0.0806343}),
                (2, {"Lower": -0.11944500000000001, "Upper": 0.080555}),
                (3, {"Lower": -0.11923740000000001, "Upper": 0.0807626}),
                (4, {"Lower": -0.119343, "Upper": 0.080657}),
            ]
    },
    "9": {
        "input":
            [
                (0, {"Lower": 0.44373142947667665, "Upper": 0.6437314294766766}),
                (1, {"Lower": 0.1947548781775292, "Upper": 0.3947548781775292}),
                (2, {"Lower": 0.3308061998984888, "Upper": 0.5308061998984888}),
                (3, {"Lower": 0.12522146217891103, "Upper": 0.325221462178911}),
                (4, {"Lower": 0.058102982184994295, "Upper": 0.2581029821849943}),
            ],
        "output":
            [
                (0, {"Lower": -0.1201359, "Upper": 0.07986410000000001}),
                (1, {"Lower": -0.1193657, "Upper": 0.0806343}),
                (2, {"Lower": -0.11944500000000001, "Upper": 0.080555}),
                (3, {"Lower": -0.11923740000000001, "Upper": 0.0807626}),
                (4, {"Lower": -0.119343, "Upper": 0.080657}),
            ]
    },
    "10": {
        "input":
            [
                (0, {"Lower": -0.33132453154932007, "Upper": -0.13132453154932008}),
                (1, {"Lower": 0.1763041426054339, "Upper": 0.3763041426054339}),
                (2, {"Lower": -0.025500302609073905, "Upper": 0.1744996973909261}),
                (3, {"Lower": -0.0969044272297114, "Upper": 0.10309557277028861}),
                (4, {"Lower": -0.5137022408640275, "Upper": -0.31370224086402754}),
            ],
        "output":
            [
                (0, {"Lower": -0.1201359, "Upper": 0.07986410000000001}),
                (1, {"Lower": -0.1193657, "Upper": 0.0806343}),
                (2, {"Lower": -0.11944500000000001, "Upper": 0.080555}),
                (3, {"Lower": -0.11923740000000001, "Upper": 0.0807626}),
                (4, {"Lower": -0.119343, "Upper": 0.080657}),
            ]
    },
    "11": {
        "input":
            [
                (0, {"Lower": 0.15789938863784383, "Upper": 0.3578993886378439}),
                (1, {"Lower": 0.008023928339643688, "Upper": 0.2080239283396437}),
                (2, {"Lower": 0.2284353323163489, "Upper": 0.4284353323163489}),
                (3, {"Lower": 0.35854642976566986, "Upper": 0.5585464297656698}),
                (4, {"Lower": 0.12570919966037516, "Upper": 0.32570919966037515}),
            ],
        "output":
            [
                (0, {"Lower": -0.1201359, "Upper": 0.07986410000000001}),
                (1, {"Lower": -0.1193657, "Upper": 0.0806343}),
                (2, {"Lower": -0.11944500000000001, "Upper": 0.080555}),
                (3, {"Lower": -0.11923740000000001, "Upper": 0.0807626}),
                (4, {"Lower": -0.119343, "Upper": 0.080657}),
            ]
    },
    "12": {
        "input":
            [
                (0, {"Lower": -0.0989937251346115, "Upper": 0.10100627486538852}),
                (1, {"Lower": -0.0802916365423613, "Upper": 0.11970836345763872}),
                (2, {"Lower": -0.23260346042087063, "Upper": -0.03260346042087062}),
                (3, {"Lower": -0.35837592763285553, "Upper": -0.15837592763285555}),
                (4, {"Lower": -0.3646802725640277, "Upper": -0.16468027256402765}),
            ],
        "output":
            [
                (0, {"Lower": -0.12011068490420883, "Upper": 0.07988931509579118}),
                (1, {"Lower": -0.11917224612114324, "Upper": 0.08082775387885677}),
                (2, {"Lower": -0.11908712865305213, "Upper": 0.08091287134694788}),
                (3, {"Lower": -0.1190048172914437, "Upper": 0.08099518270855631}),
                (4, {"Lower": -0.11887367491113227, "Upper": 0.08112632508886775}),
            ]
    },
    "13": {
        "input":
            [
                (0, {"Lower": 0.03885141656763072, "Upper": 0.23885141656763073}),
                (1, {"Lower": 0.16658572348933795, "Upper": 0.36658572348933793}),
                (2, {"Lower": -0.06162235651130368, "Upper": 0.13837764348869633}),
                (3, {"Lower": 0.1940949095786265, "Upper": 0.3940949095786265}),
                (4, {"Lower": -0.18542035241408697, "Upper": 0.014579647585913041}),
            ],
        "output":
            [
                (0, {"Lower": -0.1201359, "Upper": 0.07986410000000001}),
                (1, {"Lower": -0.1193657, "Upper": 0.0806343}),
                (2, {"Lower": -0.11944500000000001, "Upper": 0.080555}),
                (3, {"Lower": -0.11923740000000001, "Upper": 0.0807626}),
                (4, {"Lower": -0.119343, "Upper": 0.080657}),
            ]
    },
    "14": {
        "input":
            [
                (0, {"Lower": -0.0733720116648898, "Upper": 0.12662798833511021}),
                (1, {"Lower": -0.22886373579429523, "Upper": -0.028863735794295214}),
                (2, {"Lower": 0.31466283583593724, "Upper": 0.5146628358359372}),
                (3, {"Lower": -0.3499881443064524, "Upper": -0.14998814430645238}),
                (4, {"Lower": -0.44098474925640374, "Upper": -0.24098474925640376}),
            ],
        "output":
            [
                (0, {"Lower": -0.1201359, "Upper": 0.07986410000000001}),
                (1, {"Lower": -0.1193657, "Upper": 0.0806343}),
                (2, {"Lower": -0.11944500000000001, "Upper": 0.080555}),
                (3, {"Lower": -0.11923740000000001, "Upper": 0.0807626}),
                (4, {"Lower": -0.119343, "Upper": 0.080657}),
            ]
    },
    "15": {
        "input":
            [
                (0, {"Lower": 0.3927162531243882, "Upper": 0.5927162531243882}),
                (1, {"Lower": -0.24937829100007394, "Upper": -0.04937829100007393}),
                (2, {"Lower": -0.33991246294062083, "Upper": -0.13991246294062085}),
                (3, {"Lower": -0.16091215679744494, "Upper": 0.03908784320255507}),
                (4, {"Lower": -0.5625971613018141, "Upper": -0.3625971613018142}),
            ],
        "output":
            [
                (0, {"Lower": -0.1201359, "Upper": 0.07986410000000001}),
                (1, {"Lower": -0.1193657, "Upper": 0.0806343}),
                (2, {"Lower": -0.11944500000000001, "Upper": 0.080555}),
                (3, {"Lower": -0.11923740000000001, "Upper": 0.0807626}),
                (4, {"Lower": -0.119343, "Upper": 0.080657}),
            ]
    },
    "16": {
        "input":
            [
                (0, {"Lower": -0.08840513164819316, "Upper": 0.11159486835180685}),
                (1, {"Lower": 0.3070403863789337, "Upper": 0.5070403863789337}),
                (2, {"Lower": -0.3793480349538433, "Upper": -0.17934803495384324}),
                (3, {"Lower": -0.2849310466410917, "Upper": -0.08493104664109166}),
                (4, {"Lower": -0.03805166128319562, "Upper": 0.1619483387168044}),
            ],
        "output":
            [
                (0, {"Lower": -0.1201359, "Upper": 0.07986410000000001}),
                (1, {"Lower": -0.1193657, "Upper": 0.0806343}),
                (2, {"Lower": -0.11944500000000001, "Upper": 0.080555}),
                (3, {"Lower": -0.11923740000000001, "Upper": 0.0807626}),
                (4, {"Lower": -0.119343, "Upper": 0.080657}),
            ]
    },
    "17": {
        "input":
            [
                (0, {"Lower": 0.37800991661094263, "Upper": 0.5780099166109427}),
                (1, {"Lower": -0.5827724814796779, "Upper": -0.3827724814796779}),
                (2, {"Lower": 0.17855182597906347, "Upper": 0.37855182597906345}),
                (3, {"Lower": 0.008179202913771916, "Upper": 0.20817920291377193}),
                (4, {"Lower": -0.478378810334688, "Upper": -0.2783788103346879}),
            ],
        "output":
            [
                (0, {"Lower": -0.1201359, "Upper": 0.07986410000000001}),
                (1, {"Lower": -0.1193657, "Upper": 0.0806343}),
                (2, {"Lower": -0.11944500000000001, "Upper": 0.080555}),
                (3, {"Lower": -0.11923740000000001, "Upper": 0.0807626}),
                (4, {"Lower": -0.119343, "Upper": 0.080657}),
            ]
    },
    "18": {
        "input":
            [
                (0, {"Lower": 0.4127656939065426, "Upper": 0.6127656939065426}),
                (1, {"Lower": -0.15090008678579733, "Upper": 0.04909991321420268}),
                (2, {"Lower": 0.09131321375348597, "Upper": 0.29131321375348596}),
                (3, {"Lower": -0.31561373609417465, "Upper": -0.11561373609417466}),
                (4, {"Lower": 0.14763048135094067, "Upper": 0.34763048135094066}),
            ],
        "output":
            [
                (0, {"Lower": -0.1201359, "Upper": 0.07986410000000001}),
                (1, {"Lower": -0.1193657, "Upper": 0.0806343}),
                (2, {"Lower": -0.11944500000000001, "Upper": 0.080555}),
                (3, {"Lower": -0.11923740000000001, "Upper": 0.0807626}),
                (4, {"Lower": -0.119343, "Upper": 0.080657}),
            ]
    },
    "19": {
        "input":
            [
                (0, {"Lower": 0.3177557120609673, "Upper": 0.5177557120609674}),
                (1, {"Lower": 0.33771741604304484, "Upper": 0.5377174160430448}),
                (2, {"Lower": -0.21198980330407566, "Upper": -0.011989803304075647}),
                (3, {"Lower": -0.372058295477231, "Upper": -0.17205829547723103}),
                (4, {"Lower": -0.2049507383934189, "Upper": -0.004950738393418891}),
            ],
        "output":
            [
                (0, {"Lower": -0.1201359, "Upper": 0.07986410000000001}),
                (1, {"Lower": -0.1193657, "Upper": 0.0806343}),
                (2, {"Lower": -0.11944500000000001, "Upper": 0.080555}),
                (3, {"Lower": -0.11923740000000001, "Upper": 0.0807626}),
                (4, {"Lower": -0.119343, "Upper": 0.080657}),
            ]
    },
    "medium": {
        "input":
            [
                (0, {"Lower": 1.1, "Upper": 1.3}),
                (1, {"Lower": 4.5, "Upper": 4.65}),
                (2, {"Lower": -0.5, "Upper": -0.1}),
                (3, {"Lower": 0.45, "Upper": 0.5}),
                (4, {"Lower": -1.65, "Upper": -1.5}),
            ],
        "output":
            [
                (0, {"Lower": 1.0}),
                (1, {"Lower": 4.0}),
                (2, {"Lower": 1.0}),
                (3, {"Lower": 2.5}),
                (4, {"Lower": .1})
            ]
    },
    "long-medium": {
        "input":
            [
                # long-medium property
                (0, {"Lower": 1.0, "Upper": 1.5}),
                (1, {"Lower": 4.5, "Upper": 4.75}),
                (2, {"Lower": -0.5, "Upper": -0.0}),
                (3, {"Lower": 0.45, "Upper": 0.55}),
                (4, {"Lower": -1.65, "Upper": -1.35}),
            ],
        "output":
            [
                (0, {"Lower": 1.0}),
                (1, {"Lower": 4.0}),
                (2, {"Lower": 1.0}),
                (3, {"Lower": 2.5}),
                (4, {"Lower": .1})
            ]
    },
    "long": {
        "input":
            [
                # long property (property 1)
                (0, {"Lower": 0.6, "Upper": 0.6798577687}),
                (1, {"Lower": -0.5, "Upper": 0.5}),
                (2, {"Lower": -0.5, "Upper": 0.5}),
                (3, {"Lower": 0.45, "Upper": 0.5}),
                (4, {"Lower": -0.5, "Upper": -0.45}),
            ],
        "output":
            [
                (0, {"Lower": 1.0}),
                (1, {"Lower": 4.0}),
                (2, {"Lower": 1.0}),
                (3, {"Lower": 2.5}),
                (4, {"Lower": .1})
            ]
    },
    "default": {
        "input":
            [
                # (0, {"Lower": 0.3, "Upper": 0.35}),
                # (1, {"Lower": 0.7, "Upper": 0.91}),
                # (2, {"Lower": -0.1, "Upper": -0.06}),
                # (3, {"Lower": 0.045, "Upper": 0.05}),
                # (4, {"Lower": -0.05, "Upper": 1.45}),
                # (0, {"Lower": 0.06, "Upper": 0.06798577687}),
                # (1, {"Lower": -0.01, "Upper": 0.01}),
                # (2, {"Lower": -0.01, "Upper": 0.01}),
                # (3, {"Lower": 0.045, "Upper": 0.05}),
                # (4, {"Lower": -0.05, "Upper": -0.045}),
                # long property (property 1)
                # (0, {"Lower": 0.6, "Upper": 0.6798577687}),
                # (1, {"Lower": -0.5, "Upper": 0.5}),
                # (2, {"Lower": -0.5, "Upper": 0.5}),
                # (3, {"Lower": 0.45, "Upper": 0.5}),
                # (4, {"Lower": -0.5, "Upper": -0.45}),
                # long-medium property
                # (0, {"Lower": 1.0, "Upper": 1.5}),
                # (1, {"Lower": 4.5, "Upper": 4.75}),
                # (2, {"Lower": -0.5, "Upper": -0.0}),
                # (3, {"Lower": 0.45, "Upper": 0.55}),
                # (4, {"Lower": -1.65, "Upper": -1.35}),
                # medium property
                (0, {"Lower": 1.1, "Upper": 1.3}),
                (1, {"Lower": 4.5, "Upper": 4.65}),
                (2, {"Lower": -0.5, "Upper": -0.1}),
                (3, {"Lower": 0.45, "Upper": 0.5}),
                (4, {"Lower": -1.65, "Upper": -1.5}),
            ],
        "output":
            [
                # (0, {"Lower": 3.9911256459}),
                # (1, {"Lower": 3.9911256459}),
                # (2, {"Lower": 3.9911256459}),
                # (3, {"Lower": 3.9911256459}),
                # (4, {"Lower": 3.9911256459})
                (0, {"Lower": 1.0}),
                (1, {"Lower": 4.0}),
                (2, {"Lower": 1.0}),
                (3, {"Lower": 2.5}),
                (4, {"Lower": .1})
            ]
    }
}


def is_adversarial_satisfying_assignment(network, test_property, output) -> bool:
    target_label = test_property.get("_adversarial_target_label")
    non_target_labels = test_property.get("_adversarial_non_target_labels", [])
    output_property = test_property["output"]
    violation_operator = test_property.get("_adversarial_violation_operator", "ge")

    if target_label is not None and len(output) > len(output_property):
        target_value = output[int(target_label)]
        if violation_operator == "le":
            fallback_threshold = (
                output_property[0][1].get("Upper", 0.0)
                if output_property
                else 0.0
            )
        else:
            fallback_threshold = (
                output_property[0][1].get("Lower", 0.0)
                if output_property
                else 0.0
            )
        default_threshold = float(test_property.get("_adversarial_threshold", fallback_threshold))
        if len(output_property) == len(non_target_labels):
            label_specs = list(zip(non_target_labels, output_property))
        else:
            label_specs = [
                (non_target_label, (None, {"Lower": default_threshold, "Upper": default_threshold}))
                for non_target_label in non_target_labels
            ]
        if violation_operator == "le":
            return any(
                output[int(non_target_label)] - target_value <= float(bounds.get("Upper", default_threshold))
                for non_target_label, (_, bounds) in label_specs
            )
        return any(
            output[int(non_target_label)] - target_value >= float(bounds.get("Lower", default_threshold))
            for non_target_label, (_, bounds) in label_specs
        )

    if violation_operator == "le":
        return any(
            output[int(variable)] <= bounds.get("Upper", 0.0)
            for variable, bounds in output_property
        )
    return any(
        output[int(variable)] >= bounds.get("Lower", 0.0)
        for variable, bounds in output_property
    )


def is_satisfying_assignment(network, test_property, output, variables2nodes) -> bool:
    """
    returns if output is valid a.t. test_property output bounds
    @output - dict from node_name to value
    @test_property - see get_query() method documentation

    # for the reader, e.g:
    output
    {'x_7_0': 2.4388628607018603,
     'x_7_1': 0.1540985927922452,
     'x_7_2': 0.44519896688589594,
     'x_7_3': 1.0787132557294057,
     'x_7_4': 0.49820921502029597}

    output_vars
    [(605, 'x_7_0'),
     (606, 'x_7_1'),
     (607, 'x_7_2'),
     (608, 'x_7_3'),
     (609, 'x_7_4')]

    test_property["output"][0]
    (0, {'Upper': 3.9911256459})

    test_property["output"][0][0]
    0

    output_vars[test_property["output"][0][0]]
    (605, 'x_7_0')

    output_vars[test_property["output"][0][0]][1]
    'x_7_0'

    output[output_vars[test_property["output"][0][0]][1]]
    2.4388628607018603
    """
    if test_property.get("type") == "adversarial":
        return is_adversarial_satisfying_assignment(
            network=network,
            test_property=test_property,
            output=output,
        )

    output_property = test_property["output"]
    # print(f"output_property={output_property}")
    sorted_vars = sorted(variables2nodes.items(), key=lambda x: x[0])
    # print(f"sorted_vars={sorted_vars}")
    # couples of (variable index, node name) of output layer only
    output_vars = sorted_vars[-len(network.layers[-1].nodes):]
    # print(f"output_vars={output_vars}")
    for variable, bounds in output_property:
        index_name = output_vars[variable]
        # print(f"index_name={index_name}")
        node_name = index_name[1]
        # print(f"node_name={node_name}")
        if "Lower" in bounds.keys():
            # print("Lower in bounds.keys")
            # print(node_name)
            lower_bound = bounds["Lower"]
            # print("output")
            print(output[int(node_name.replace("_inc", "").split("_")[-1])])
            # print("lower")
            # print(lower_bound)
            if lower_bound > output[int(node_name.replace("_inc", "").split("_")[-1])]:
                return False
        if "Upper" in bounds.keys():
            upper_bound = bounds["Upper"]
            if output[int(node_name.replace("_inc", "").split("_")[-1])] > upper_bound:
                return False
    return True


def _read_property_payload(property_filename) -> Dict:
    with open(property_filename) as pf:
        raw_text = pf.read().strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(raw_text)
    except (SyntaxError, ValueError):
        return ast.literal_eval("{" + raw_text.rstrip(",") + "}")


def read_test_property(property_filename, property_id: str = None) -> Dict:
    test_property = _read_property_payload(property_filename)
    if property_id is not None and property_id in test_property:
        return test_property[property_id]
    return test_property


def read_test_properties(property_filename) -> Dict:
    return _read_property_payload(property_filename)


def write_test_property(test_property, filename) -> None:
    with open(filename, "w") as pf:
        pf.write(json.dumps(test_property))


def get_test_property_acas(property_id: str = consts.PROPERTY_ID) -> Dict:
    test_property = TEST_PROPERTY_ACAS.get(property_id, None)
    if test_property is None:
        raise Exception(f"wrong property_id: {property_id}")
    return test_property


def get_test_property_tiny() -> Dict:
    test_property_tiny = {
        "input":
            [
                (0, {"Lower": 0.6, "Upper": 0.6798577687}),
                (1, {"Lower": -0.1, "Upper": 0.1}),
                (2, {"Lower": -0.1, "Upper": 0.1}),
                (3, {"Lower": 0.45, "Upper": 0.5}),
                (4, {"Lower": -0.5, "Upper": -0.45}),
            ],
        "output":
            [
                (0, {"Lower": 250.0}),
                (1, {"Lower": 250.0}),
                (2, {"Lower": 250.0}),
                (3, {"Lower": 250.0}),
                (4, {"Lower": 250.0})
            ]
    }
    return test_property_tiny


def get_test_property_input_2_output_1() -> Dict:
    """
    :return: test property of random net of order 2,2,1
    """
    test_property_tiny = {
        "input":
            [
                (0, {"Lower": 0.6, "Upper": 0.6798577687}),
                (1, {"Lower": -0.1, "Upper": 0.1}),
            ],
        "output":
            [
                (0, {"Lower": 250.0}),
            ]
    }
    return test_property_tiny


def get_test_property_input_3_output_1() -> Dict:
    """
    :return: test property of random net of order 2,2,1
    """
    test_property_tiny = {
        "input":
            [
                (0, {"Lower": 0.6, "Upper": 0.6798577687}),
                (1, {"Lower": -0.1, "Upper": 0.1}),
                (2, {"Lower": -0.1, "Upper": 0.1}),
            ],
        "output":
            [
                (0, {"Lower": 250.0}),
            ]
    }
    return test_property_tiny


def get_winner_and_runner_up(adversarial_test_property):
    """
    in acas, the minimal value is chosen as answer
    this function check which nodes are the winner (minimal) and runner_up (the
    second minimal) a.t. specific test_property
    :param adversarial_test_property: test_property
    :return: values+indices of two minimal output nodes
    """
    # print(adversarial_test_property["output"])
    s = []
    for index, d_val in adversarial_test_property["output"]:
        s.append((index, d_val["Lower"]))
    winner, runner = float("inf"), float("inf")
    w_ind, r_ind = None, None
    for (x, y) in s:
        if winner > y:
            winner = y
            w_ind = x
    for (x, y) in s:
        if x == w_ind:
            continue
        if runner > y:
            runner = y
            r_ind = x
    return winner, w_ind, runner, r_ind


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--property_filename", dest="filename", default=consts.PROPERY_FILNAME_ACAS)
    parser.add_argument("-rw", "--read_write", dest="read_write", default="r", choices=["r", "w"],
                        help="read property from file or write property to file")
    args = parser.parse_args()
    if args.read_write == "r":
        prop = read_test_property(args.filename)
        print(prop)
    else:  # "w"
        prop = get_test_property_acas()
        write_test_property(test_property=prop, filename=args.filename)
