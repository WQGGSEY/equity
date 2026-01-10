class Strategy:
    def __init__(self, name="BaseStrategy"):
        self.name = name
        self.md = None # MarketData
        
    def initialize(self, market_data):
        self.md = market_data
        
    def on_bar(self, date, universe_tickers, portfolio):
        """
        [Must Implement]
        매일 호출되는 함수.
        :param date: 현재 날짜 (Timestamp)
        :param universe_tickers: 오늘 유니버스에 포함된 티커 리스트
        :param portfolio: 현재 포트폴리오 상태
        :return: list of orders [{'ticker':..., 'action':..., 'quantity':...}]
        """
        raise NotImplementedError