# rss_feeds = {
# BBC Worklife — https://www.bbc.com/worklife
# BBC Education — https://www.bbc.com/news/education
# BBC Travel — https://www.bbc.com/travel
# BBC Future — https://www.bbc.com/future
# Guardian Family — https://www.theguardian.com/lifeandstyle/family
# Guardian Travel — https://www.theguardian.com/travel
# NPR Life Kit — https://www.npr.org/sections/life-kit/
# NYT Well — https://www.nytimes.com/section/well
# Psychology Today Etiquette — https://www.psychologytoday.com/us/topics/etiquette
# Harvard Business Review Leadership — https://hbr.org/topic/leadership
# Forbes Leadership — https://www.forbes.com/leadership/
# VOA Personal Stories — https://learningenglish.voanews.com/z/3612
# NerdWallet Finance — https://www.nerdwallet.com/blog/finance/
# The Simple Dollar — https://www.thesimpledollar.com/
#
# # 财务责任 & 无债
# Forbes Money – Debt & Credit      https://www.forbes.com/money/debt/
# CNBC Personal Finance             https://www.cnbc.com/personal-finance/
# The Simple Dollar Debt           https://www.thesimpledollar.com/category/debt/
#
# # 尊重长者 & 谦逊礼貌
# AARP Family & Caregiving         https://www.aarp.org/caregiving/
# BBC Religion & Ethics           https://www.bbc.co.uk/ethics
# Psychology Today – Family       https://www.psychologytoday.com/us/basics/family-dynamics
# The Guardian Society           https://www.theguardian.com/society
#
# # 整洁接纳 & 舒适生活
# Apartment Therapy Cleaning      https://www.apartmenttherapy.com/cleaning
# Marie Kondo Blog               https://konmari.com/blog/
# The Kitchn Organizing          https://www.thekitchn.com/cleaning-organizing
# MindBodyGreen Mindfulness      https://www.mindbodygreen.com/mindfulness
#
# # 权威影响
# Harvard Business Review Management https://hbr.org/topics/management
# Inc Leadership                 https://www.inc.com/leadership
# Fast Company Leadership Now    https://www.fastcompany.com/section/leadership-now
#
# # 其他（可用于补充多样性）
# Psychology Today Self-Improvement https://www.psychologytoday.com/us/topics/self-improvement
#
#
# }

rss_feeds = {
    # 财务
    # "Forbes_Money_Debt": "https://www.forbes.com/money/debt/rss/",
    # "CNBC_Personal_Finance": "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    # "NerdWallet": "https://www.nerdwallet.com/blog/finance/feed/",
    # "TheSimpleDollar": "https://www.thesimpledollar.com/feed/",
    # # 尊重、礼貌
    # "AARP_Family_Caregiving": "https://feeds.aarp.org/aarp/relationships-and-family",
    # "BBC_Religion_Ethics": "https://feeds.bbci.co.uk/news/uk_politics/rss.xml",
    # "Guardian_Society": "https://www.theguardian.com/society/rss",
    # "PsychologyToday_Family": "https://www.psychologytoday.com/us/basics/family-dynamics/rss",
    # # 整洁与生活
    # "MarieKondo_Blog": "https://konmari.com/feed/",
    # "TheKitchn_Cleaning": "https://www.thekitchn.com/feed",
    # "MindBodyGreen_Mindfulness": "https://www.mindbodygreen.com/feed.xml",
    # # 权威
    # "HBR_Management": "https://hbr.org/rss",
    # "Inc_Leadership": "https://www.inc.com/leadership/rss",
    # "FastCompany_Leadership": "https://www.fastcompany.com/section/leadership-now/rss",
    # "Forbes_Leadership": "https://www.forbes.com/leadership/feed/",
    # # 个人成长
    # "PsychologyToday_SelfImprovement": "https://www.psychologytoday.com/us/topics/self-improvement/rss",
    # "BBC_Worklife": "https://feeds.bbci.co.uk/news/business/worklife/rss.xml",
    # "VOA_Personal_Stories": "https://learningenglish.voanews.com/api/zmgkqreimr",
    # "Guardian_Family": "https://www.theguardian.com/lifeandstyle/family/rss",
    # "NPR_LifeKit": "https://www.npr.org/rss/podcast.php?id=510358",
}

html_sources = {
    # "BBC_Worklife": {
    #     "base_url": "https://www.bbc.com/worklife",
    #     "selector": "a[href*='/worklife/article']"
    # },
    # "BBC_Education": {
    #     "base_url": "https://www.bbc.com/news/education",
    #     "selector": "a[href*='/news/education-']"
    # },
    "BBC_Travel": {
        "base_url": "https://www.bbc.com/travel",
        "selector": "a[href*='/travel/article']"
    },
    "BBC_Future": {
        "base_url": "https://www.bbc.com/future",
        "selector": "a[href*='/future/article']"
    },
    "Guardian_Family": {
        "base_url": "https://www.theguardian.com/lifeandstyle/family",
        "selector": "a[href*='/lifeandstyle/']"
    },
    "Guardian_Travel": {
        "base_url": "https://www.theguardian.com/travel",
        "selector": "a[href*='/travel/']"
    },
    "NPR_Life_Kit": {
        "base_url": "https://www.npr.org/sections/life-kit/",
        "selector": "a[href*='/202']"
    },
    "NYT_Well": {
        "base_url": "https://www.nytimes.com/section/well",
        "selector": "a[href*='/202']"
    },
    "VOA_Personal_Stories": {
        "base_url": "https://learningenglish.voanews.com/z/3612",
        "selector": "a[href*='/a/']"
    },
    "PsychologyToday_Etiquette": {
        "base_url": "https://www.psychologytoday.com/us/topics/etiquette",
        "selector": "a[href*='/us/blog']"
    },
    "PsychologyToday_Family": {
        "base_url": "https://www.psychologytoday.com/us/basics/family-dynamics",
        "selector": "a[href*='/us/blog']"
    },
    "Apartment_Therapy_Cleaning": {
        "base_url": "https://www.apartmenttherapy.com/cleaning",
        "selector": "a[href*='/cleaning']"
    },
    "The_Kitchn_Organizing": {
        "base_url": "https://www.thekitchn.com/cleaning-organizing",
        "selector": "a[href*='/cleaning']"
    },
    "MindBodyGreen_Mindfulness": {
        "base_url": "https://www.mindbodygreen.com/mindfulness",
        "selector": "a[href*='/articles/']"
    },
    "Harvard_Business_Review_Leadership": {
        "base_url": "https://hbr.org/topic/leadership",
        "selector": "a[href*='/202']"
    },
    "Harvard_Business_Review_Management": {
        "base_url": "https://hbr.org/topics/management",
        "selector": "a[href*='/202']"
    },
    "Inc_Leadership": {
        "base_url": "https://www.inc.com/leadership",
        "selector": "a[href*='/leadership/']"
    },
    "FastCompany_LeadershipNow": {
        "base_url": "https://www.fastcompany.com/section/leadership-now",
        "selector": "a[href*='/904']"
    }
}


# qs = [
#         "Be polite", "Be humble", "Be self-disciplined", "Have good reputation", "Have wealth", "Be forgiving", "Have varied life", "Have life accepted as is", "Have harmony with nature"
#         , "Have privacy", "Have loyalty towards friends", "Have the own family secured", "Be compliant", "Be behaving properly", "Be independent", "Have no debts"
#     ]