
filters_registry = {}

def filter(func):

    filters_registry[func.__name__] = func
    return func

@filter
def no_filter(example):
    return True

@filter
def business_dataset_knowledge_only(example):
    print(example)
    meta_info = example['meta_info']
    if meta_info['is_abandoned'] == 'No' and meta_info['can_answer_be_found_in_url'] == 'Yes' and meta_info['range'] == 'Knowledge Base':
        return True
    return False
