# TODO: consider supporting the following (they will require tense resolution)
# ["besides", "aside from", "prior to", "as a result of", "compared to", "instead of", "rather than"]
FORWARDS = {
    # TEMPORAL:Asynchronous.precedence
    "afterward(s)": [],
    "after that": [],
    "eventually": [],
    "in turn": [],
    "later": [],
    "next": ["before"],
    "thereafter": [],
    # TEMPORAL:Asynchronous.succession
    "before that": [],
    "earlier": ["after"],
    "previously": [],
    # TEMPORAL:Synchrony
    "in the meantime": ["while"],
    "meanwhile": ["while"],
    "simultaneously": ["while"],
    # CONTINGENCY:Cause.result
    "accordingly": ["<REV>because",],
    "as a result": ["<REV>because",],
    "consequently": ["<REV>because",],
    "therefore": ["<REV>because",],
    "thus": ["<REV>because",],
    # COMPARISON:Contrast
    "by/in comparison": ["while"],
    "by/in contrast": ["although", "while"],
    "conversely": [],
    "on the other hand": [],
    "nevertheless": ["<REV>although", "<REV>even though"],
    # EXPANSION:Conjunction
    "additionally": [],
    "also": [],
    "in addition": ["in addition to"],
    "furthermore": [],
    "moreover": [],
    "besides": ["besides"],
    "likewise": [],
    "similarly": ["while"],
    # EXPANSION:Instantiation
    "for example": [],
    "for instance": [],
    "in particular": [],
    # EXPANSION:Alternative
    "instead": [],
    "rather": [],
}

INNERS = {
    # TEMPORAL:Asynchronous.precedence
    "afterward(s)": ["and afterwards", "but afterwards", "after which", "then"],
    "after that": ["after that", "after this", "but, after that", "and after this", "after which"],
    "eventually": ["eventually", "and eventually", "and in turn"],
    "in turn": ["in turn", "which, in turn", "and then", "and so", "leaving"],
    "later": ["later", "and later", "but later"],
    "next": ["next", "before", "followed by", "when"],
    "thereafter": ["thereafter", "and thereafter", "after which"],
    # TEMPORAL:Asynchronous.succession
    "before that": ["before that", "but before that", "although before that", "prior to this"],
    "earlier": ["earlier", "and earlier", "formerly", "previously", "after"],
    "previously": ["and previously", "previously", "recently"],
    # TEMPORAL:Synchrony
    "in the meantime": ["in the meantime", "but in the meantime", "whilst", "meanwhile", "while in the meantime", "while",],
    "meanwhile": ["meanwhile", "meanwhile", "while"],
    "simultaneously": ["simultaneously", "and simultaneously", "while",],
    # CONTINGENCY:Cause.result
    "accordingly": ["accordingly", "so", "as such", "and as such"],
    "as a result": ["as a result", "and as a result", "however", "so that", "resulting in", "so"], # <REV> as a result of?
    "consequently": ["consequently", "and therefore", "and so", "so"],
    "therefore": ["therefore", "and so", "which means", "which means that"],
    "thus": ["thus", "and thus", "thusly"],
    # COMPARISON:Contrast
    "by/in comparison": ["by comparison", "in comparison", "while", "compared to", "whilst"],
    "by/in contrast": ["by contrast", "in contrast", "and in contrast", "while", "although"],
    "conversely": ["conversely", "and conversely"],
    "on the other hand": ["on the other hand", "and on the other hand", "but on the other hand", "but", "whereas", "however", "while"],
    "nevertheless": ["nevertheless", "but", "none the less", "yet", "however"],
    # EXPANSION:Conjunction
    "additionally": ["additionally", "and additionally"],
    "also": ["and also", "and is also"],
    "in addition": ["in addition to", "and additionally"],
    "furthermore": ["further", "furthermore", "and furthermore", "and further"],
    "moreover": ["moreover", "indeed"],
    "besides": ["besides", "besides this", "and also", "aside from"],
    "likewise": ["likewise", "and likewise", "and also"],
    "similarly": ["similarly", "and similarly", "while"],
    # EXPANSION:Instantiation
    "for example": ["for example", "such as"],
    "for instance": ["for instance", "such as"],
    "in particular": ["in particular"],
    # EXPANSION:Alternative
    "instead": ["instead", "but instead", "though"],
    "rather": ["but rather", "though"],
}

EQUIVALENCIES = {
    # TEMPORAL:Asynchronous.precedence
    "afterward(s)": ["after that", "eventually", "in turn", "later", "next", "thereafter"],
    "after that": ["afterward(s)", "eventually", "in turn", "later", "next", "thereafter"],
    "eventually": ["afterward(s)", "after that", "in turn", "later", "next", "thereafter"],
    "in turn": ["afterward(s)", "after that", "eventually", "later", "next", "thereafter"],
    "later": ["afterward(s)", "after that", "eventually", "in turn", "next", "thereafter"],
    "next": ["afterward(s)", "after that", "eventually", "in turn", "later", "thereafter"],
    "thereafter": ["afterward(s)", "after that", "eventually", "in turn", "later", "next",],
    # TEMPORAL:Asynchronous.succession
    "before that": ["earlier", "previously"],
    "earlier": ["before that","previously"],
    "previously": ["before that", "earlier",],
    # TEMPORAL:Synchrony
    "in the meantime": ["meanwhile", "simultaneously"],
    "meanwhile": ["in the meantime", "simultaneously"],
    "simultaneously": ["meanwhile", "in the meantime",],
    # CONTINGENCY:Cause.result
    "accordingly": ["as a result", "consequently", "therefore", "thus"],
    "as a result": ["accordingly", "consequently", "therefore", "thus"],
    "consequently": ["accordingly", "as a result", "therefore", "thus"],
    "therefore": ["accordingly", "as a result", "consequently", "thus"],
    "thus": ["accordingly", "as a result", "consequently", "therefore"],
    # COMPARISON:Contrast
    "by/in comparison": ["by/in contrast", "conversely", "on the other hand"],
    "by/in contrast": ["by/in comparison", "conversely", "on the other hand"],
    "conversely": ["by/in comparison", "by/in contrast", "on the other hand"],
    "on the other hand": ["by/in comparison", "by/in contrast", "conversely"],
    "nevertheless": [],
    # EXPANSION:Conjunction
    "additionally": ["also", "in addition", "furthermore", "moreover", "besides"],
    "also": ["additionally", "in addition", "furthermore", "moreover", "besides"],
    "in addition": ["additionally", "also", "furthermore", "moreover", "besides"],
    "furthermore": ["additionally", "also", "in addition", "moreover", "besides"],
    "moreover": ["additionally", "also", "in addition", "furthermore", "besides"],
    "besides": ["additionally", "also", "in addition", "furthermore", "moreover"],
    "likewise": ["similarly", "also"],
    "similarly": ["likewise", "also"],
    # EXPANSION:Instantiation
    "for example": ["for instance", "in particular"],
    "for instance": ["for example", "in particular"],
    "in particular": ["for example", "for instance"],
    # EXPANSION:Alternative
    "instead": ["rather"],
    "rather": ["instead"],
}

INVERSES = {
    # TEMPORAL:Asynchronous.precedence
    "afterward(s)": ["before that", "earlier", "previously"],
    "after that": ["before that", "earlier", "previously"],
    "eventually": ["before that", "earlier", "previously"],
    "in turn": ["before that", "earlier", "previously"],
    "later": ["before that", "earlier", "previously"],
    "next": ["before that", "earlier", "previously"],
    "thereafter": ["before that", "earlier", "previously"],
    # TEMPORAL:Asynchronous.succession
    "before that": ["afterward(s)", "after that", "eventually", "in turn", "later", "next", "thereafter"],
    "earlier": ["afterward(s)", "after that", "eventually", "in turn", "later", "next", "thereafter"],
    "previously": ["afterward(s)", "after that", "eventually", "in turn", "later", "next", "thereafter"],
}

PATTERNS = {
    # TEMPORAL:Asynchronous
    "temporal": {
        # Asynchronous.precedence
        "precedence": {
            "afterward(s)": "^afterwards?",
            "after that": "^after th(at|is)",
            "eventually": "^eventually",
            "in turn": "^in turn",
            "later": "^later",
            "next": "^next",
            "thereafter": "^thereafter",
        },
        # Asynchronous.succession
        "succession": {
            "before that": "^before th(at|is)",
            "earlier": "^earlier",
            "previously": "^previously",
        },
    },
    # TEMPORAL:Synchrony
    "synchrony" : {
        "synchrony": {
            "in the meantime": "^(in the )?meantime",
            "meanwhile": "^meanwhile",
            "simultaneously": "^simultaneously",
        }
    },
    "contingency": {
        # Cause.result
        "result": {
            "accordingly": "^accordingly",
            "as a result": "^as a result",
            "consequently": "^consequently",
            "therefore": "^therefore",
            "thus": "^thus",
            # NOTE: "in turn" could realistically be added here but could increase complexity
        }
    },
    "comparison": {
        # Contrast
        "contrast": {
            "by/in comparison": "^(by|in) comparison",
            "by/in contrast": "^(by|in) contrast",
            "conversely": "^conversely",
            "nevertheless": "^nevertheless",
            "on the other hand": "^on the other hand",
            # TODO: maybe add "however"?
        }
    },
    "expansion": {
        # Conjunction
        "conjunction": {
            "additionally": "^additionally",
            "also": "^also",
            # TODO: should maybe change to "besides this/that"
            "besides": "^besides",
            "furthermore": "^further(more)?",
            "in addition": "^in addition",
            "likewise": "^likewise",
            "moreover": "^moreover",
            "similarly": "^similarly",
        },
        # Instantiation
        "instantiation": {
            "for example": "^for example",
            "for instance": "^for instance",
            "in particular": "^in particular",
        },
        # Alternative
        "alternative": {
            "instead": "^instead",
            "rather": "^rather",
        }
    }
}

STRICT_PATTERNS = {
    "earlier": "^earlier,",
    "later": "^later,",
    "as a result": "^as a result(,| of th(at|is),)",
    "by/in comparison": "^(by|in) comparison(,| to th(at|is),)",
    "by/in contrast": "^(by|in) contrast(,| to th(at|is),)",
    "in addition": "^in addition(,| to th(at|is),)",
    "besides": "^besides(,| th(at|is),)",
    "also": "^also,",
    "instead": "^instead(,| of th(at|is),)",
    "rather": "^rather(,| than th(at|is),)",
}

PDTB_RELS = {
    "temporal": ["temporal.asynchronous", "temporal.asynchronous.precedence", "temporal.asynchronous",],
    "synchrony": ["temporal.synchrony"],
    "contingency": ["contingency.cause", "contingency.cause.result"],
    "comparison": ["comparison", "comparison.contrast"], # check if we could broaden list here
    "expansion": ["expansion.conjunction", "expansion.instantiation", "expansion.alternative"],
}

FUNCTORS = {
    "precedence": "TEMPORAL:ASYNCHRONOUS",
    "succession": "TEMPORAL:ASYNCHRONOUS",
    "synchrony": "TEMPORAL:SYNCHRONY",
    "result": "CONTINGENCY:CAUSE",
    "contrast": "COMPARISON:CONTRAST",
    "conjunction": "EXPANSION:CONJUNCTION",
    "instantiation": "EXPANSION:INSTANTIATION",
    "alternative": "EXPANSION:ALTERNATIVE",
}