from typing import List, Tuple
import re

rules_def: List[Tuple[str, str]] = [
    ('x', 'ks'),
    ('sh', 'sz'),
    ('zż', 'z-ż'),
    ('ww', 'w_w'),
    ('odziem', 'od_ziem'),
    ('...', '…'),
    ('arzn', 'ar_zn'),
    ('zinte', 'z_inte'),
    (' bruce', ' brus'),
    (' hulk', ' halk'),
    ('she-hulk', 'szi-halk'),
    ('lady-hulk', 'lejdi-halk'),
    ('girl-hulk', 'gerl-halk'),
    (' walters', ' łolters'),
    (' steve', ' stiw'),
    (' roger', ' rodżer'),
    ('jennifer', 'dżenifer'),
    (' jen ', ' dżen '),
    (' yen ', ' jen '),
    ('cheetos', 'czitos'),
    (' holiway', 'holiway'),
    (' blonsky', 'blonski'),
    ('carring', 'karing'),
    ('izaac', 'izaak'),
    ('sprite', 'sprajt'),
]
    
def gen_rule(rule_tuple: Tuple[str, str]) -> Tuple[re.Pattern, str]:
    from_rule, to_rule = rule_tuple
    prefix, suffix = False, False
    
    from_rule = from_rule.replace('.', '\\.')

    if from_rule[0] == ' ':
        prefix = True
    
    if from_rule[-1] == ' ':
        suffix = True
    
    _prefix = '([ \,\.\;\!\:\?]|^)' if prefix else ''
    _suffix = '([ \,\.\;\!\:\?]|$)' if suffix else ''

    _from_rule = (f'{_prefix}{from_rule.strip()}{_suffix}')

    _to_prefix = '\g<1>' if _prefix != '' else ''
    _to_suffix = '\g<1>' if _prefix == '' and _suffix != '' else '\g<2>' if _suffix != '' else ''

    _to_rule = (f'{_to_prefix}{to_rule.strip()}{_to_suffix}')

    return re.compile(_from_rule), _to_rule

rules = map(gen_rule, rules_def)

def shmart_replace(text: str) -> str:
    text = text.replace('GLKiH', 'Gie eL Ka i Ha')

    text = text[1:] if text[0] == '-' else text
    result = text.lower()

    for _, (rule_from, rule_to) in enumerate(rules):
        result = rule_from.sub(rule_to, result)
    
    return result