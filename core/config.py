MODE_ON = False

mode2act = {'social': ['social-open-close',  'social-continue',  'social', 'social-other'], \
'recommend': ['push-tailored-info-suggest', 'push-general-info-suggest', 'push-conditional-suggest', 'push-conditional'], \
'inform': ['push-general-info-inform', 'push-tailored-info-inform', 'push-conditional-inform', 'push-general-info', 'push-tailored-info', 'push-other', 'push'], \
'query': ['pull-select', 'pull-bool', 'pull-fill', 'pull-other', 'pull']}

act2mode = {'social-open-close': 'social', 'social-continue': 'social', 'social': 'social', \
            'social-other': 'social', 'push-tailored-info-suggest': 'recommend', \
            'push-general-info-suggest': 'recommend', 'push-conditional-suggest': 'recommend', \
            'push-conditional': 'recommend', 'push-general-info-inform': 'inform', \
            'push-tailored-info-inform': 'inform', 'push-conditional-inform': 'inform', \
            'push-general-info': 'inform', 'push-tailored-info': 'inform', \
            'push-other': 'inform', 'push': 'inform', 'pull-select': 'query', \
            'pull-bool': 'query', 'pull-fill': 'query', 'pull-other': 'query', 'pull': 'query'}
# modes: recommend, query, inform, social
mode2id = {'recommend': 0, 'query': 1, 'inform': 2, 'social': 3}