try:
    from .smart_search import (
        SearchConfig,
        RelevanceImprover,
        SmartSearchEngine
    )
    __all__ = ['SearchConfig', 'RelevanceImprover', 'SmartSearchEngine']
    SMART_SEARCH_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Умный поиск недоступен: {e}")
    SMART_SEARCH_AVAILABLE = False
    __all__ = []