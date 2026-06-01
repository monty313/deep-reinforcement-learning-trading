//+------------------------------------------------------------------+
//| FTMO_DQN.mq5                                                     |
//| DQN-driven FTMO EA — communicates with Python via shared files   |
//|                                                                  |
//| SETUP:                                                           |
//|  1. Start live_agent.py first (wait for "Ready" message)         |
//|  2. Attach this EA to ANY 1m chart (e.g. EURUSD.sim M1)         |
//|  3. EA trades ALL symbols in the list below                      |
//|  4. Set your account size, hard stops, spread limits in Inputs   |
//|                                                                  |
//| REQUIRES: Tools → Options → Expert Advisors                      |
//|   ✓ Allow automated trading                                      |
//|   ✓ Allow DLL imports  (for file read/write)                     |
//|                                                                  |
//| NOTE: If the Experts log says "agent_ready.txt not found" or
//|       shows a path like MT5Bridge\agent_ready.txt, then Python is
//|       writing to the wrong folder or MT5 is still running an old
//|       compiled EA copy. Recompile and reattach the EA after
//|       starting Python with the correct --bridge-dir path.
//+------------------------------------------------------------------+
#property copyright "FTMO DQN Agent"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>

//── Inputs ────────────────────────────────────────────────────────────────────
// NOTE: BridgeDir is now unused. The EA uses its own terminal's MQL5\Files sandbox.
//       Python must read/write from: C:\Users\user\AppData\Roaming\MetaQuotes\Terminal\49CDDEAA95A409ED22BD2287BB67CB9C\MQL5\Files\
input string   BridgeDir          = "MT5Bridge";         // DEPRECATED (kept for reference only)
input double   AccountSize        = 100000.0;            // FTMO account size ($)
input double   HardStopDD_Pct     = 2.0;                 // Hard stop: max daily drawdown %
input double   HardStopProfit_Pct = 3.0;                 // Hard stop: max daily profit %
input double   MaxLotSize         = 1.0;                 // Maximum lot size per trade
input double   ForexMaxSpreadPips = 3.0;                 // Forex: max spread in pips
input int      MaxTradesPerDay    = 800;                 // Hard cap on trades per day
input int      MagicNumber        = 20260101;            // EA magic number
input bool     PrintDebug         = false;               // Print debug messages

//── Symbol lists ──────────────────────────────────────────────────────────────
string ForexSymbols[] = {
    "EURUSD.sim","AUDUSD.sim","USDCAD.sim","USDCHF.sim",
    "NZDUSD.sim","AUDCAD.sim","USDSEK.sim","AUDCHF.sim",
    "CADCHF.sim","EURAUD.sim","EURCAD.sim","EURCHF.sim",
    "EURNZD.sim","EURSEK.sim","NZDCAD.sim","NZDCHF.sim",
    "GBPJPY.sim"
};
string IndexSymbols[]     = {"US100.sim","US500.sim","US30.sim"};
string CommoditySymbols[] = {"XAUUSD.sim","USOIL.sim","XAGUSD.sim"};

//── State variables ───────────────────────────────────────────────────────────
CTrade  Trade;
int     LastAction        = 0;
string  LastBarTimes[];   // per-symbol last processed bar time
double  DayStartBalance   = 0.0;
bool    TradingBlocked    = false;
int     TradesToday       = 0;
datetime LastDayReset     = 0;
int     TotalSymbols      = 0;
string  AllSymbols[];

//── File paths ────────────────────────────────────────────────────────────────
string BarFile;
string ActionFile;
string ReadyFile;

void PrintBridgeHelp(const string sandboxPath)
{
    Print("[EA] Bridge Diagnostics:");
    Print("[EA] 1) Expected Python bridge folder: ", sandboxPath);
    Print("[EA] 2) Make sure live_agent.py is running with:");
    Print("[EA]      python -m gpu_rl_trading.live.live_agent --checkpoint \"<path>\" --bridge-dir \"" + sandboxPath + "\"");
    Print("[EA] 3) Recompile this EA in MetaEditor and reattach it to the chart.");
    Print("[EA] 4) Make sure this EA is loaded from the terminal's MQL5\\Experts folder, not from a stale copy.");
    Print("[EA] 5) Do NOT use C:\\MT5Bridge for this terminal: MT5 sandbox I/O only sees its own MQL5\\Files folder.");
    Print("[EA] Common failure modes:");
    Print("[EA]  - Python is not started yet");
    Print("[EA]  - Python is using the wrong --bridge-dir folder");
    Print("[EA]  - MT5 is still running an old compiled EA copy");
    Print("[EA]  - agent_ready.txt is in C:\\MT5Bridge instead of the terminal sandbox");
}

//+------------------------------------------------------------------+
int OnInit()
{
    Trade.SetExpertMagicNumber(MagicNumber);
    Trade.SetDeviationInPoints(10);

    // Files are relative to terminal sandbox: MQL5\Files\
    // Python must write to: C:\Users\user\AppData\Roaming\MetaQuotes\Terminal\49CDDEAA95A409ED22BD2287BB67CB9C\MQL5\Files\
    BarFile    = "bar_data.csv";
    ActionFile = "action.txt";
    ReadyFile  = "agent_ready.txt";
    FolderCreate(".", 0);

    // Build full symbol list: forex + indices + commodities
    // Auto-detect actual broker symbol names (handles .sim, no suffix, .r, etc.)
    string baseForex[]     = {"EURUSD","AUDUSD","USDCAD","USDCHF","NZDUSD","AUDCAD",
                               "USDSEK","AUDCHF","CADCHF","EURAUD","EURCAD","EURCHF",
                               "EURNZD","EURSEK","NZDCAD","NZDCHF","GBPJPY"};
    string baseIndices[]   = {"US100","US500","US30"};
    string baseCommodities[]= {"XAUUSD","USOIL","XAGUSD"};

    int nF = ArraySize(baseForex);
    int nI = ArraySize(baseIndices);
    int nC = ArraySize(baseCommodities);
    int nAll = nF + nI + nC;
    ArrayResize(AllSymbols, nAll);
    ArrayResize(LastBarTimes, nAll);

    int idx = 0;
    for(int i = 0; i < nF; i++)  { AllSymbols[idx] = ResolveSymbol(baseForex[i]);      LastBarTimes[idx] = ""; idx++; }
    for(int i = 0; i < nI; i++)  { AllSymbols[idx] = ResolveSymbol(baseIndices[i]);    LastBarTimes[idx] = ""; idx++; }
    for(int i = 0; i < nC; i++)  { AllSymbols[idx] = ResolveSymbol(baseCommodities[i]); LastBarTimes[idx] = ""; idx++; }

    // remove any symbols that couldn't be resolved
    TotalSymbols = 0;
    for(int i = 0; i < nAll; i++)
    {
        if(AllSymbols[i] != "")
        {
            AllSymbols[TotalSymbols]   = AllSymbols[i];
            LastBarTimes[TotalSymbols] = "";
            TotalSymbols++;
        }
    }
    ArrayResize(AllSymbols,   TotalSymbols);
    ArrayResize(LastBarTimes, TotalSymbols);

    // Check Python agent is running
    string sandboxPath = TerminalInfoString(TERMINAL_DATA_PATH) + "\\MQL5\\Files\\";
    Print("[EA] Sandbox Files path: ", sandboxPath);
    Print("[EA] Point Python --bridge-dir to this folder");
    if(!FileIsExist(ReadyFile))
    {
        Print("[EA] WARNING: agent_ready.txt not found in the EA sandbox.");
        Print("[EA] This means Python is not writing to the terminal's MQL5\\Files folder.");
        Print("[EA] agent_ready.txt must be in: ", sandboxPath);
        PrintBridgeHelp(sandboxPath);
    }
    else
    {
        Print("[EA] Python agent is ready.");
    }

    DayStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    LastDayReset    = TimeCurrent();
    Print("[EA] Initialised. AccountSize=", AccountSize,
          "  DD_Hard=", HardStopDD_Pct, "%",
          "  Profit_Hard=", HardStopProfit_Pct, "%",
          "  MaxLot=", MaxLotSize,
          "  Symbols=", TotalSymbols);
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("[EA] Removed. Reason=", reason);
}

//+------------------------------------------------------------------+
void OnTick()
{
    // ── 1. Day reset at 00:00 CET ─────────────────────────────────
    CheckDayReset();

    // ── 2. Check hard stops ───────────────────────────────────────
    if(CheckHardStops()) return;

    // ── 3. Check Python agent is still alive ──────────────────────
    if(!FileIsExist(ReadyFile))
    {
        if(PrintDebug) Print("[EA] Python agent not ready — skipping tick");
        return;
    }

    // ── 4. For each symbol, check for new 1m bar and act ─────────
    for(int i = 0; i < TotalSymbols; i++)
    {
        string sym = AllSymbols[i];
        ProcessSymbol(sym, i);
        if(TradingBlocked) break;
    }
}

//+------------------------------------------------------------------+
void ProcessSymbol(string sym, int symIdx)
{
    if(TradesToday >= MaxTradesPerDay)
    {
        if(PrintDebug) Print("[EA] Max trades reached for today (", MaxTradesPerDay, ")");
        return;
    }

    // Get current 1m bar
    MqlRates rates[];
    if(CopyRates(sym, PERIOD_M1, 0, 2, rates) < 2) return;

    // Only act on a new completed bar (bar[1] is the just-closed bar)
    string barTime = TimeToString(rates[1].time, TIME_DATE|TIME_MINUTES);
    if(barTime == LastBarTimes[symIdx]) return;   // already processed this bar for this symbol

    // ── Spread filter ─────────────────────────────────────────────
    if(!SpreadOK(sym, rates[1].close)) return;

    // ── Write bar data for Python ─────────────────────────────────
    WriteBarFile(sym, rates[1]);

    // ── Wait briefly for Python to write action ───────────────────
    Sleep(80);

    // ── Read action from Python ───────────────────────────────────
    int action = ReadAction();
    LastBarTimes[symIdx] = barTime;

    if(PrintDebug)
        Print("[EA] ", sym, " bar=", barTime, "  action=", action);

    // ── Execute action ────────────────────────────────────────────
    ExecuteAction(sym, action, rates[1].close);
}

//+------------------------------------------------------------------+
string ResolveSymbol(string base)
{
    // Try exact name first, then common suffixes
    string suffixes[] = {"", ".sim", ".r", "m", ".pro", ".ecn"};
    for(int i = 0; i < ArraySize(suffixes); i++)
    {
        string candidate = base + suffixes[i];
        if(SymbolInfoInteger(candidate, SYMBOL_SELECT) || SymbolSelect(candidate, true))
            return candidate;
    }
    Print("[EA] WARNING: Could not resolve symbol: ", base, " — skipping");
    return "";
}

//+------------------------------------------------------------------+
bool SpreadOK(string sym, double lastClose)
{
    double ask    = SymbolInfoDouble(sym, SYMBOL_ASK);
    double bid    = SymbolInfoDouble(sym, SYMBOL_BID);
    double spread = ask - bid;

    if(IsForex(sym))
    {
        // forex: spread in pips (1 pip = 0.0001 for most pairs, 0.01 for JPY pairs)
        double pipSize = (StringFind(sym, "JPY") >= 0) ? 0.01 : 0.0001;
        double spreadPips = spread / pipSize;
        if(spreadPips > ForexMaxSpreadPips)
        {
            if(PrintDebug)
                Print("[EA] ", sym, " spread=", spreadPips, " pips > max ", ForexMaxSpreadPips, " — skipping");
            return false;
        }
    }
    else
    {
        // indices/commodities: spread must be less than the last 1m close price
        // (catches runaway spreads on illiquid instruments)
        if(spread >= lastClose)
        {
            if(PrintDebug)
                Print("[EA] ", sym, " spread=", spread, " >= close=", lastClose, " — skipping");
            return false;
        }
    }
    return true;
}

//+------------------------------------------------------------------+
bool IsForex(string sym)
{
    for(int i = 0; i < ArraySize(ForexSymbols); i++)
        if(ForexSymbols[i] == sym) return true;
    return false;
}

//+------------------------------------------------------------------+
void WriteBarFile(string sym, MqlRates &bar)
{
    string barTime = TimeToString(bar.time, TIME_DATE|TIME_MINUTES);
    string line = sym + "," + barTime + "," +
                  DoubleToString(bar.open,  5) + "," +
                  DoubleToString(bar.high,  5) + "," +
                  DoubleToString(bar.low,   5) + "," +
                  DoubleToString(bar.close, 5) + "," +
                  DoubleToString((double)bar.tick_volume, 1);

    int handle = FileOpen(BarFile, FILE_WRITE|FILE_TXT|FILE_ANSI);
    if(handle == INVALID_HANDLE)
    {
        Print("[EA] ERROR writing bar file: ", GetLastError());
        Print("[EA] Check that the terminal sandbox folder exists and this EA is attached to the correct MT5 terminal.");
        return;
    }
    FileWriteString(handle, line);
    FileClose(handle);
}

//+------------------------------------------------------------------+
int ReadAction()
{
    if(!FileIsExist(ActionFile))
    {
        if(PrintDebug)
            Print("[EA] Note: action.txt not found. Python may not have written a response yet or may be using the wrong bridge folder.");
        return 0;
    }

    int handle = FileOpen(ActionFile, FILE_READ|FILE_TXT|FILE_ANSI);
    if(handle == INVALID_HANDLE) return 0;

    string content = FileReadString(handle);
    FileClose(handle);

    int action = (int)StringToInteger(content);
    if(action < 0 || action > 6) action = 0;
    return action;
}

//+------------------------------------------------------------------+
void ExecuteAction(string sym, int action, double close)
{
    // action meanings:
    //  0 = FLAT   (close any position, don't open)
    //  1 = BUY_S  (buy small)
    //  2 = BUY_M  (buy medium)
    //  3 = BUY_L  (buy large)
    //  4 = SELL_S (sell small)
    //  5 = SELL_M (sell medium)
    //  6 = SELL_L (sell large)

    int    posType  = GetPositionType(sym);  // 1=long, -1=short, 0=flat
    int    newSide  = ActionSide(action);    // 1=buy, -1=sell, 0=flat

    // close existing position on reversal or FLAT signal
    if(posType != 0 && (newSide == 0 || newSide != posType))
    {
        ClosePosition(sym);
        TradesToday++;
    }

    // open new position
    if(newSide != 0 && GetPositionType(sym) == 0)
    {
        double lots = CalcLots(sym, action, close);
        if(lots <= 0) return;

        if(newSide == 1)
            Trade.Buy(lots, sym, 0, 0, 0, "DQN_BUY");
        else
            Trade.Sell(lots, sym, 0, 0, 0, "DQN_SELL");

        if(Trade.ResultRetcode() == TRADE_RETCODE_DONE)
        {
            TradesToday++;
            if(PrintDebug)
                Print("[EA] ", sym, " OPEN ", (newSide==1?"BUY":"SELL"),
                      " lots=", lots, " trades_today=", TradesToday);
        }
        else
        {
            Print("[EA] ", sym, " order failed: ", Trade.ResultRetcodeDescription());
        }
    }
}

//+------------------------------------------------------------------+
double CalcLots(string sym, int action, double close)
{
    // size multipliers matching training: small=0.25, med=0.50, large=1.00
    double sizeMult[];
    ArrayResize(sizeMult, 7);
    sizeMult[0]=0.0; sizeMult[1]=0.25; sizeMult[2]=0.50; sizeMult[3]=1.00;
    sizeMult[4]=0.25; sizeMult[5]=0.50; sizeMult[6]=1.00;

    double balance    = AccountInfoDouble(ACCOUNT_BALANCE);
    double targetEq   = DayStartBalance * (1.0 + 0.025);   // 2.5% daily target
    double gapDollars = MathMax(targetEq - balance, 0.0);

    // ATR proxy: use (high - low) of last bar as rough ATR
    MqlRates rates[];
    double atrPrice = 0.0005;   // safe default
    if(CopyRates(sym, PERIOD_M1, 1, 14, rates) == 14)
    {
        double sumTR = 0;
        for(int i = 0; i < 14; i++)
            sumTR += rates[i].high - rates[i].low;
        atrPrice = MathMax(sumTR / 14.0, 1e-6);
    }

    double tickVal  = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_VALUE);
    double tickSize = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_SIZE);
    double atrDollars = (tickSize > 0) ? (atrPrice / tickSize) * tickVal : atrPrice * 100000.0;
    atrDollars = MathMax(atrDollars, 1e-6);

    double targetLots = gapDollars / atrDollars;

    // when gap already closed, size conservatively off DD headroom
    if(gapDollars <= 0)
    {
        double ddHeadroom = MathMax(0.0, (HardStopDD_Pct / 100.0) * balance
                                        - MathMax(0.0, DayStartBalance - balance));
        targetLots = (ddHeadroom / atrDollars) * 0.5;
    }

    double lots = targetLots * sizeMult[action];

    // clamp to broker limits and our max
    double minLot  = SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN);
    double maxLot  = MathMin(MaxLotSize, SymbolInfoDouble(sym, SYMBOL_VOLUME_MAX));
    double lotStep = SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP);

    lots = MathMax(lots, minLot);
    lots = MathMin(lots, maxLot);

    // clamp to available free margin (never use more than 10% of free margin per trade)
    double freeMargin = AccountInfoDouble(ACCOUNT_FREEMARGIN);
    double marginPer1Lot = 0;
    if(!OrderCalcMargin(ORDER_TYPE_SELL, sym, 1.0,
                        SymbolInfoDouble(sym, SYMBOL_BID), marginPer1Lot))
        marginPer1Lot = freeMargin;  // fallback: assume 1 lot uses all margin
    if(marginPer1Lot > 0)
    {
        double maxAffordableLots = (freeMargin * 0.10) / marginPer1Lot;
        lots = MathMin(lots, maxAffordableLots);
    }

    lots = MathMax(lots, minLot);
    // round to lot step
    if(lotStep > 0)
        lots = MathFloor(lots / lotStep) * lotStep;

    if(lots < minLot) return 0;  // can't afford even min lot
    return lots;
}

//+------------------------------------------------------------------+
int ActionSide(int action)
{
    if(action >= 1 && action <= 3) return  1;   // buy
    if(action >= 4 && action <= 6) return -1;   // sell
    return 0;                                    // flat
}

//+------------------------------------------------------------------+
int GetPositionType(string sym)
{
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if(PositionSelectByTicket(ticket))
        {
            if(PositionGetString(POSITION_SYMBOL) == sym &&
               PositionGetInteger(POSITION_MAGIC) == MagicNumber)
            {
                int type = (int)PositionGetInteger(POSITION_TYPE);
                return (type == POSITION_TYPE_BUY) ? 1 : -1;
            }
        }
    }
    return 0;
}

//+------------------------------------------------------------------+
void ClosePosition(string sym)
{
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if(PositionSelectByTicket(ticket))
        {
            if(PositionGetString(POSITION_SYMBOL) == sym &&
               PositionGetInteger(POSITION_MAGIC) == MagicNumber)
            {
                Trade.PositionClose(ticket);
            }
        }
    }
}

//+------------------------------------------------------------------+
void CloseAllPositions()
{
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if(PositionSelectByTicket(ticket))
        {
            if(PositionGetInteger(POSITION_MAGIC) == MagicNumber)
                Trade.PositionClose(ticket);
        }
    }
}

//+------------------------------------------------------------------+
bool CheckHardStops()
{
    if(TradingBlocked) return true;

    double balance  = AccountInfoDouble(ACCOUNT_BALANCE);
    double equity   = AccountInfoDouble(ACCOUNT_EQUITY);

    // use equity for DD check (includes open positions)
    double ddPct     = (DayStartBalance > 0)
                       ? (DayStartBalance - equity) / DayStartBalance * 100.0
                       : 0.0;
    double profitPct = (DayStartBalance > 0)
                       ? (equity - DayStartBalance) / DayStartBalance * 100.0
                       : 0.0;

    if(ddPct >= HardStopDD_Pct)
    {
        Print("[EA] *** HARD STOP: Daily DD ", DoubleToString(ddPct, 2),
              "% >= ", HardStopDD_Pct, "% — closing all positions ***");
        CloseAllPositions();
        TradingBlocked = true;
        return true;
    }

    if(profitPct >= HardStopProfit_Pct)
    {
        Print("[EA] *** HARD STOP: Daily profit ", DoubleToString(profitPct, 2),
              "% >= ", HardStopProfit_Pct, "% — closing all positions ***");
        CloseAllPositions();
        TradingBlocked = true;
        return true;
    }

    return false;
}

//+------------------------------------------------------------------+
void CheckDayReset()
{
    // FTMO day resets at 00:00 CET (UTC+1 standard, UTC+2 summer)
    // MT5 server time is typically already in CET/CEST for OANDA
    // Reset when the day changes on server time
    datetime now     = TimeCurrent();
    MqlDateTime nowDT, lastDT;
    TimeToStruct(now,          nowDT);
    TimeToStruct(LastDayReset, lastDT);

    if(nowDT.day != lastDT.day)
    {
        double prevBalance = DayStartBalance;
        DayStartBalance    = AccountInfoDouble(ACCOUNT_BALANCE);
        TradingBlocked     = false;
        TradesToday        = 0;
        LastDayReset       = now;

        Print("[EA] *** DAY RESET ***  prev_balance=", prevBalance,
              "  new_balance=", DayStartBalance,
              "  time=", TimeToString(now, TIME_DATE|TIME_MINUTES));
    }
}
//+------------------------------------------------------------------+
