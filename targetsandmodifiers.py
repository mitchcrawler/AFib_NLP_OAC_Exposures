import os
import csv
import re

class ModifiersAndTargets(object):
    def __init__(self):
        self.modifiers = []
        self.targets = []
        
    def addTarget(self, name, regex, direction='forward', target_type="TARGET"):
        self.targets.append({
            'Lex' : name,
            'Type' : target_type,
            'Regex' : regex,
            'Direction' : direction
        })
        
    def addModifier(self, name, regex, modifier_type="NEGATED_EXISTENCE", direction='forward'):
        self.modifiers.append({
            'Lex' : name,
            'Type' : modifier_type,
            'Regex' : regex,
            'Direction' : direction
        })
        
    def testText(self, text):
        """
        Searches the text for all of the regular expressions in targets and modifiers and returns a list of the 
        names of the matching modifiers and/or targets.
        """
        
        target_matches = []
        modifier_matches = []
        
        for collection in [self.modifiers, self.targets]:
            for item in collection:
                if re.search(item['Regex'], text):
                    if item['Type'] == 'TARGET':
                        target_matches.append(item)
                    else:
                        modifier_matches.append(item)
                    
        target_names = list(map(lambda x: x['Lex'], target_matches))
        modifier_names = list(map(lambda x: x['Lex'], modifier_matches))
        return target_names, modifier_names
        
    def writeTargetsAndModifiers(self, path, targets_name='targets.tsv', modifiers_name='modifiers.tsv'):
        if not os.path.isdir(path):
            raise RuntimeError(f"The specified path is not a directory. Path: {path}")
        
        fieldnames = ['Lex', 'Type', 'Regex', 'Direction']
        
        with open(os.path.join(path, targets_name), 'w') as target_file, open(os.path.join(path ,modifiers_name), 'w') as modifiers_file:
            for handle, obj_list in [(target_file, self.targets), (modifiers_file, self.modifiers)]:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter='\t')
                writer.writeheader()
                writer.writerows(obj_list)