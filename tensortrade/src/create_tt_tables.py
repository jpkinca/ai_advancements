import os
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from dotenv import load_dotenv

# Load credentials from .env or use direct connection string
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL") or "postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway"

Base = declarative_base()

class tt_episode(Base):
    __tablename__ = "tt_episode"
    id = Column(Integer, primary_key=True)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    stop_reason = Column(String)
    max_drawdown = Column(Float)
    sharpe_ratio = Column(Float)
    turnover = Column(Float)

class tt_portfolio(Base):
    __tablename__ = "tt_portfolio"
    id = Column(Integer, primary_key=True)
    episode_id = Column(Integer, ForeignKey("tt_episode.id"))
    timestamp = Column(DateTime)
    net_worth = Column(Float)
    cash_balance = Column(Float)
    episode = relationship("tt_episode")

class tt_holdings(Base):
    __tablename__ = "tt_holdings"
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey("tt_portfolio.id"))
    instrument = Column(String)
    quantity = Column(Float)
    portfolio = relationship("tt_portfolio")

class tt_prices(Base):
    __tablename__ = "tt_prices"
    id = Column(Integer, primary_key=True)
    instrument = Column(String)
    timestamp = Column(DateTime)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

class tt_action(Base):
    __tablename__ = "tt_action"
    id = Column(Integer, primary_key=True)
    episode_id = Column(Integer, ForeignKey("tt_episode.id"))
    timestamp = Column(DateTime)
    instrument = Column(String)
    action_value = Column(Float)
    episode = relationship("tt_episode")

class tt_order(Base):
    __tablename__ = "tt_order"
    id = Column(Integer, primary_key=True)
    episode_id = Column(Integer, ForeignKey("tt_episode.id"))
    timestamp = Column(DateTime)
    instrument = Column(String)
    order_type = Column(String)
    size = Column(Float)
    price = Column(Float)
    episode = relationship("tt_episode")

class tt_reward(Base):
    __tablename__ = "tt_reward"
    id = Column(Integer, primary_key=True)
    episode_id = Column(Integer, ForeignKey("tt_episode.id"))
    timestamp = Column(DateTime)
    reward_value = Column(Float)
    episode = relationship("tt_episode")

class tt_observation(Base):
    __tablename__ = "tt_observation"
    id = Column(Integer, primary_key=True)
    episode_id = Column(Integer, ForeignKey("tt_episode.id"))
    timestamp = Column(DateTime)
    features = Column(Text)  # Store as JSON string if needed
    episode = relationship("tt_episode")

if __name__ == "__main__":
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)
    print("All tt_ tables created successfully.")
