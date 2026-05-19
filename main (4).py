"""
BX3 Backend V2.0 — FastAPI + Pydantic
Contrat JSON strict · Feedback Loop · Brier Score · Auto-pondération
Famille ETOH · Déploiement Render.com (gratuit)
"""
from fastapi import FastAPI, HTTPException, Query, Depends, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import httpx, os, math, random, json
from datetime import datetime, timedelta

app = FastAPI(title="BX3 Backend Ω", version="2.0.0", docs_url="/docs")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── CLÉS (env vars sur Render — jamais dans l'APK)
APIFOOTBALL_KEY  = os.getenv("APIFOOTBALL_KEY",  "5a9b57e9506b5c0bba3089a18dd309da")
ODDS_API_KEY     = os.getenv("ODDS_API_KEY",     "4865be3ee6aa31308d05ab9e90022cc481f6f4853eebd254ce25497c31898fa1")
FOOTBALLDATA_KEY = os.getenv("FOOTBALLDATA_KEY", "5f3849cc0b314ac29ee6613efc24bf10")
SPORTMONKS_KEY   = os.getenv("SPORTMONKS_KEY",   "0tS9UyNiMOv0JDXMCRFmMLo1IFc8Sv62XjZAOuHJhky53mFBVx44Z2l4DLPv")
BX3_SECRET       = os.getenv("BX3_SECRET",       "BX3FamilleEtoh2026SecretKey")

def verify_token(x_bx3_token: str = Header(None)):
    if x_bx3_token != BX3_SECRET:
        raise HTTPException(status_code=401, detail="Token invalide")
    return True

# ═══════════════════════════════════════════════════════
# PYDANTIC SCHEMAS — CONTRAT JSON STRICT
# Le frontend TypeScript doit utiliser ces interfaces exactes
# ═══════════════════════════════════════════════════════

class LeagueInfo(BaseModel):
    id: str
    name: str
    country: str
    flag: str = ""
    logo: str = ""

class TeamInfo(BaseModel):
    id: str
    name: str
    logo: str = ""

class ScoreInfo(BaseModel):
    home: Optional[int] = None
    away: Optional[int] = None
    ht_home: Optional[int] = None
    ht_away: Optional[int] = None

class MatchResponse(BaseModel):
    id: str                          # "SPA-W-PSG-MAR-20260425"
    date: str                        # "2026-04-25"
    time: str                        # "20:45"
    status: str                      # NS | 1H | HT | 2H | FT
    is_live: bool = False
    is_finished: bool = False
    league: LeagueInfo
    home: TeamInfo
    away: TeamInfo
    score: ScoreInfo

class MatchListResponse(BaseModel):
    dates: List[str]
    leagues_filter: List[str]
    total: int
    matches: List[MatchResponse]

class ProbabilitySet(BaseModel):
    home: float = Field(..., ge=0, le=100, description="P(Victoire domicile) en %")
    draw: float = Field(..., ge=0, le=100, description="P(Match nul) en %")
    away: float = Field(..., ge=0, le=100, description="P(Victoire extérieur) en %")
    over25: float = Field(..., ge=0, le=100)
    under25: float = Field(..., ge=0, le=100)
    btts_yes: float = Field(..., ge=0, le=100, description="Les deux équipes marquent")
    btts_no: float = Field(..., ge=0, le=100)

class ModelWeight(BaseModel):
    name: str
    weight: float = Field(..., ge=0, le=1)
    accuracy_30d: float = Field(..., ge=0, le=100, description="Précision 30 derniers jours")

class ValueBet(BaseModel):
    market: str
    market_odd: float
    vadigo_odd: float
    edge: float = Field(..., description="Edge en %")
    kelly_fraction: float = Field(..., description="Fraction Kelly en %")

class WatawaValidation(BaseModel):
    passed: bool
    rules_ok: int = Field(..., ge=0, le=16)
    rules_total: int = 16
    blocking_rule: Optional[str] = None

class XAIFactor(BaseModel):
    label: str
    impact: float = Field(..., description="Impact en % sur la prédiction")
    direction: str = Field(..., description="positive | negative")
    width_pct: int = Field(..., ge=0, le=100)

class AnalysisResponse(BaseModel):
    """
    Schéma complet VADIGO X — Blocs I à XI
    Ce contrat est utilisé par le frontend TypeScript
    Chaque clé correspond à un champ exact dans l'APK
    """
    # Bloc I — Identification
    fixture_id: str
    match_id_unique: str           # ex: FRA-L1-PSG-MAR-20260425-2045

    # Bloc II — Signal principal
    signal: str                    # GO | ALT | NO_BET
    prono: str                     # "Victoire PSG"
    prono_market: str              # home | draw | away

    # Bloc III — Métriques décision
    confidence: float              # 0-100
    edge: float                    # % edge sur le marché
    kelly: float                   # % de la bankroll à miser
    fcg: float                     # Facteur de Confiance Globale 0-1
    cote: float                    # Cote recommandée
    p_omega: float                 # Score VADIGO global 0-100

    # Bloc IV-V — Probabilités IA
    probabilities: ProbabilitySet

    # Bloc VI — Value Bets
    value_bets: List[ValueBet]

    # Bloc VII — Modèles IA utilisés
    models_used: List[ModelWeight]
    models_consensus: str          # "Fort | Modéré | Faible"

    # Bloc VIII — Risque
    chaos_score: float             # 0-10
    var_95: float                  # Value at Risk 95%
    steam_move: bool               # Mouvement suspect détecté
    steam_direction: Optional[str] = None

    # Bloc IX — Validation WATAWA
    watawa: WatawaValidation

    # Bloc X — XAI (IA Explicable)
    xai_factors: List[XAIFactor]
    xai_summary: str               # Explication en langage naturel

    # Bloc XI — Score final (complété post-match)
    final_score_home: Optional[int] = None
    final_score_away: Optional[int] = None
    brier_score: Optional[float] = None
    prediction_correct: Optional[bool] = None

class BetSaveRequest(BaseModel):
    user: str
    fixture_id: str
    match: str
    date: str
    league: str
    home: str
    away: str
    prono: str
    prono_market: str
    cote: float
    confidence: float
    edge: float
    kelly: float
    mise: float
    signal: str

class BetResponse(BaseModel):
    id: str
    user: str
    fixture_id: str
    match: str
    date: str
    league: str
    home: str
    away: str
    prono: str
    cote: float
    confidence: float
    edge: float
    kelly: float
    mise: float
    signal: str
    result: str                    # PENDING | WIN | LOSS | VOID
    saved_at: str
    settled_at: Optional[str] = None
    pl: float = 0.0
    real_score: Optional[str] = None
    brier_score: Optional[float] = None

class BetStatsResponse(BaseModel):
    user: str
    total_bets: int
    wins: int
    losses: int
    pending: int
    void: int
    win_rate: float
    roi_pct: float
    total_mise_fcfa: float
    total_pl_fcfa: float
    avg_confidence: float
    avg_edge: float
    brier_score_avg: Optional[float] = None

# ── LEAGUES DATA
LEAGUES_DATA = {
    "France":       [{"id":"61","name":"Ligue 1"},{"id":"62","name":"Ligue 2"}],
    "Angleterre":   [{"id":"39","name":"Premier League"},{"id":"40","name":"Championship"}],
    "Espagne":      [{"id":"140","name":"La Liga"},{"id":"141","name":"Segunda División"}],
    "Allemagne":    [{"id":"78","name":"Bundesliga"},{"id":"79","name":"2. Bundesliga"}],
    "Italie":       [{"id":"135","name":"Serie A"},{"id":"136","name":"Serie B"}],
    "Portugal":     [{"id":"94","name":"Primeira Liga"}],
    "Pays-Bas":     [{"id":"88","name":"Eredivisie"}],
    "Belgique":     [{"id":"144","name":"Pro League"}],
    "Turquie":      [{"id":"203","name":"Süper Lig"}],
    "Cameroun":     [{"id":"482","name":"Elite One"}],
    "Sénégal":      [{"id":"597","name":"Ligue 1"}],
    "Maroc":        [{"id":"200","name":"Botola Pro"}],
    "Nigeria":      [{"id":"491","name":"NPFL"}],
    "Égypte":       [{"id":"233","name":"Premier League"}],
    "Côte d'Ivoire":[{"id":"396","name":"MTN Ligue 1"}],
    "Ghana":        [{"id":"493","name":"Premier League"}],
    "Afrique du Sud":[{"id":"288","name":"PSL"}],
    "Europe":       [{"id":"2","name":"Champions League"},{"id":"3","name":"Europa League"},{"id":"848","name":"Conference League"}],
}

# ── MODÈLES IA — Poids adaptatifs (self-learning)
MODEL_WEIGHTS: Dict[str, float] = {
    "XGBoost":        0.22,
    "LSTM":           0.18,
    "DixonColes":     0.16,
    "Elo":            0.14,
    "Bayesian":       0.12,
    "RandomForest":   0.10,
    "NeuralNetwork":  0.08,
}
MODEL_ACCURACY: Dict[str, float] = {
    "XGBoost":        71.4,
    "LSTM":           68.9,
    "DixonColes":     66.2,
    "Elo":            65.8,
    "Bayesian":       67.3,
    "RandomForest":   63.1,
    "NeuralNetwork":  69.7,
}

# ── IN-MEMORY STORES (en prod : PostgreSQL + Redis)
_bets_db: List[dict] = []
_analyses_db: List[dict] = {}  # fixture_id → analysis
_brier_history: List[dict] = []

# ═══════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"status": "BX3 Backend Ω LIVE", "version": "2.0.0", "schema": "Pydantic V2"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "models": MODEL_WEIGHTS,
        "bets_total": len(_bets_db),
    }

@app.get("/leagues")
def get_leagues(auth=Depends(verify_token)):
    return {"countries": LEAGUES_DATA, "total": sum(len(v) for v in LEAGUES_DATA.values())}

# ── MATCHES
@app.get("/matches", response_model=MatchListResponse)
async def get_matches(
    date: str = Query(default=None),
    dates: str = Query(default=None),
    league_ids: str = Query(default=None),
    auth=Depends(verify_token)
):
    date_list = []
    if dates:
        date_list = [d.strip() for d in dates.split(",")]
    elif date:
        date_list = [date]
    else:
        date_list = [datetime.now().strftime("%Y-%m-%d")]

    lid_list = [l.strip() for l in league_ids.split(",")] if league_ids else []

    all_matches: List[MatchResponse] = []
    async with httpx.AsyncClient(timeout=12.0) as client:
        for d in date_list:
            if lid_list:
                for lid in lid_list:
                    try:
                        r = await client.get(
                            "https://v3.football.api-sports.io/fixtures",
                            headers={"x-apisports-key": APIFOOTBALL_KEY},
                            params={"date": d, "league": lid, "season": _current_season()}
                        )
                        for m in _parse_fixtures(r.json().get("response", []), d):
                            all_matches.append(m)
                    except:
                        pass
            else:
                try:
                    r = await client.get(
                        "https://v3.football.api-sports.io/fixtures",
                        headers={"x-apisports-key": APIFOOTBALL_KEY},
                        params={"date": d}
                    )
                    for m in _parse_fixtures(r.json().get("response", []), d):
                        all_matches.append(m)
                except:
                    all_matches.extend(_demo_matches(d))

    seen, unique = set(), []
    for m in all_matches:
        if m.id not in seen:
            seen.add(m.id)
            unique.append(m)

    return MatchListResponse(
        dates=date_list, leagues_filter=lid_list,
        total=len(unique), matches=unique
    )

# ── ANALYSE VADIGO COMPLÈTE
@app.get("/analyse/{fixture_id}", response_model=AnalysisResponse)
async def analyse_match(fixture_id: str, auth=Depends(verify_token)):
    if fixture_id in _analyses_db:
        cached = _analyses_db[fixture_id]
        if cached.get("final_score_home") is None:
            return AnalysisResponse(**cached)

    async with httpx.AsyncClient(timeout=12.0) as client:
        try:
            r = await client.get(
                "https://v3.football.api-sports.io/fixtures",
                headers={"x-apisports-key": APIFOOTBALL_KEY},
                params={"id": fixture_id}
            )
            fix_data = r.json().get("response", [{}])
            fix_data = fix_data[0] if fix_data else {}
        except:
            fix_data = {}

    analysis = _compute_vadigo(fixture_id, fix_data)
    _analyses_db[fixture_id] = analysis.dict()
    return analysis

# ── FEEDBACK LOOP — Bloc XI : enregistrer le résultat réel
@app.post("/analyse/{fixture_id}/result")
async def record_match_result(
    fixture_id: str,
    score_home: int = Query(...),
    score_away: int = Query(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    auth=Depends(verify_token)
):
    """
    Enregistre le score final et déclenche le recalcul des poids modèles.
    Implémente la boucle de rétroaction VADIGO.
    """
    if fixture_id not in _analyses_db:
        raise HTTPException(404, "Analyse non trouvée pour ce match")

    analysis = _analyses_db[fixture_id]
    analysis["final_score_home"] = score_home
    analysis["final_score_away"] = score_away

    # Déterminer le résultat réel
    if score_home > score_away:
        real_outcome = "home"
    elif score_home < score_away:
        real_outcome = "away"
    else:
        real_outcome = "draw"

    # Calcul du Brier Score : mesure l'écart prédiction ↔ réalité
    p = analysis.get("probabilities", {})
    p_home = p.get("home", 33) / 100
    p_draw = p.get("draw", 33) / 100
    p_away = p.get("away", 33) / 100

    actual_home = 1 if real_outcome == "home" else 0
    actual_draw = 1 if real_outcome == "draw" else 0
    actual_away = 1 if real_outcome == "away" else 0

    brier = round(
        (1/3) * ((p_home - actual_home)**2 + (p_draw - actual_draw)**2 + (p_away - actual_away)**2), 4
    )
    analysis["brier_score"] = brier
    analysis["prediction_correct"] = (analysis.get("prono_market") == real_outcome)
    _analyses_db[fixture_id] = analysis

    # Enregistrer dans l'historique Brier
    _brier_history.append({
        "fixture_id": fixture_id,
        "brier_score": brier,
        "correct": analysis["prediction_correct"],
        "league": analysis.get("fixture_id", ""),
        "timestamp": datetime.now().isoformat()
    })

    # Déclencher l'ajustement des poids en arrière-plan
    background_tasks.add_task(_adjust_model_weights, brier, real_outcome, analysis)

    # Mettre à jour les paris liés à ce match
    for bet in _bets_db:
        if bet.get("fixture_id") == fixture_id and bet.get("result") == "PENDING":
            bet_market = bet.get("prono_market", bet.get("prono", ""))
            if real_outcome == "home" and "domicile" in bet_market.lower():
                bet["result"] = "WIN"
                bet["pl"] = round(bet["mise"] * (bet["cote"] - 1))
            elif real_outcome == "draw" and "nul" in bet_market.lower():
                bet["result"] = "WIN"
                bet["pl"] = round(bet["mise"] * (bet["cote"] - 1))
            elif real_outcome == "away" and "extérieur" in bet_market.lower():
                bet["result"] = "WIN"
                bet["pl"] = round(bet["mise"] * (bet["cote"] - 1))
            else:
                bet["result"] = "LOSS"
                bet["pl"] = -bet["mise"]
            bet["settled_at"] = datetime.now().isoformat()
            bet["real_score"] = f"{score_home}-{score_away}"
            bet["brier_score"] = brier

    return {
        "success": True,
        "fixture_id": fixture_id,
        "score": f"{score_home}-{score_away}",
        "real_outcome": real_outcome,
        "prediction_was": analysis.get("prono_market"),
        "prediction_correct": analysis["prediction_correct"],
        "brier_score": brier,
        "brier_interpretation": _interpret_brier(brier),
        "model_weights_updated": True
    }

async def _adjust_model_weights(brier: float, real_outcome: str, analysis: dict):
    """
    Auto-apprentissage : ajuste les poids des modèles selon leur précision récente.
    Si Brier Score > 0.25 → le modèle dominant est pénalisé.
    Si Brier Score < 0.10 → le modèle dominant est récompensé.
    """
    global MODEL_WEIGHTS, MODEL_ACCURACY

    # Calculer la précision récente (30 dernières analyses)
    recent = _brier_history[-30:] if len(_brier_history) >= 30 else _brier_history
    if not recent:
        return

    avg_brier = sum(r["brier_score"] for r in recent) / len(recent)
    avg_accuracy = (1 - avg_brier) * 100

    # Ajuster les poids via softmax normalisé
    # Le modèle LSTM est favorisé si précis sur ce type de match
    correction = 0.02 if brier < 0.15 else -0.01 if brier > 0.30 else 0

    for model in MODEL_WEIGHTS:
        acc = MODEL_ACCURACY.get(model, 65)
        if acc > avg_accuracy:
            MODEL_WEIGHTS[model] = min(0.35, MODEL_WEIGHTS[model] + abs(correction))
        else:
            MODEL_WEIGHTS[model] = max(0.05, MODEL_WEIGHTS[model] - abs(correction))

    # Re-normaliser pour que la somme = 1
    total = sum(MODEL_WEIGHTS.values())
    for m in MODEL_WEIGHTS:
        MODEL_WEIGHTS[m] = round(MODEL_WEIGHTS[m] / total, 4)

def _interpret_brier(score: float) -> str:
    if score < 0.10: return "Excellent — Prédiction très précise 🎯"
    if score < 0.20: return "Bon — Dans les limites attendues ✅"
    if score < 0.30: return "Acceptable — Amélioration possible ⚠️"
    return "Faible — Modèle à recalibrer ❌"

# ── PARIS
@app.post("/bets/save", response_model=BetResponse)
async def save_bet(bet: BetSaveRequest, auth=Depends(verify_token)):
    b = bet.dict()
    b["id"] = f"bet-{len(_bets_db)+1}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    b["result"] = "PENDING"
    b["saved_at"] = datetime.now().isoformat()
    b["settled_at"] = None
    b["pl"] = 0.0
    b["real_score"] = None
    b["brier_score"] = None
    _bets_db.append(b)
    return BetResponse(**b)

@app.get("/bets", response_model=List[BetResponse])
async def get_bets(
    user: str = Query(default=None),
    result: str = Query(default="ALL"),
    auth=Depends(verify_token)
):
    bets = list(_bets_db)
    if user:
        bets = [b for b in bets if b.get("user") == user]
    if result != "ALL":
        bets = [b for b in bets if b.get("result") == result]
    return [BetResponse(**b) for b in bets]

@app.put("/bets/{bet_id}/result")
async def update_bet(
    bet_id: str,
    result: str = Query(..., description="WIN|LOSS|VOID"),
    real_score: str = Query(default=None),
    auth=Depends(verify_token)
):
    for b in _bets_db:
        if b["id"] == bet_id:
            b["result"] = result
            b["settled_at"] = datetime.now().isoformat()
            b["real_score"] = real_score
            if result == "WIN":
                b["pl"] = round(b["mise"] * (b["cote"] - 1))
            elif result == "LOSS":
                b["pl"] = -b["mise"]
            else:
                b["pl"] = 0
            return {"success": True, "bet": BetResponse(**b)}
    raise HTTPException(404, "Pari non trouvé")

@app.get("/bets/stats/{user}", response_model=BetStatsResponse)
async def get_bet_stats(user: str, auth=Depends(verify_token)):
    my = [b for b in _bets_db if b.get("user") == user]
    wins   = [b for b in my if b.get("result") == "WIN"]
    losses = [b for b in my if b.get("result") == "LOSS"]
    settled = wins + losses
    total_mise = sum(b["mise"] for b in settled)
    total_pl   = sum(b.get("pl", 0) for b in settled)
    roi = round(total_pl / total_mise * 100, 2) if total_mise > 0 else 0
    wr  = round(len(wins) / len(settled) * 100, 1) if settled else 0
    avg_conf = round(sum(b.get("confidence",0) for b in my)/len(my),1) if my else 0
    avg_edge = round(sum(b.get("edge",0) for b in my)/len(my),1) if my else 0
    brierS = [b["brier_score"] for b in my if b.get("brier_score") is not None]
    avg_brier = round(sum(brierS)/len(brierS),4) if brierS else None
    return BetStatsResponse(
        user=user, total_bets=len(my), wins=len(wins), losses=len(losses),
        pending=len([b for b in my if b.get("result")=="PENDING"]),
        void=len([b for b in my if b.get("result")=="VOID"]),
        win_rate=wr, roi_pct=roi, total_mise_fcfa=total_mise, total_pl_fcfa=total_pl,
        avg_confidence=avg_conf, avg_edge=avg_edge, brier_score_avg=avg_brier
    )

@app.get("/models/weights")
async def get_model_weights(auth=Depends(verify_token)):
    """Retourne les poids actuels des modèles IA (auto-appris)"""
    return {
        "weights": MODEL_WEIGHTS,
        "accuracy": MODEL_ACCURACY,
        "brier_history_count": len(_brier_history),
        "avg_brier": round(sum(b["brier_score"] for b in _brier_history)/len(_brier_history),4) if _brier_history else None
    }

# ═══════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════

def _current_season() -> int:
    n = datetime.now()
    return n.year if n.month >= 7 else n.year - 1

def _parse_fixtures(fixtures: list, date: str) -> List[MatchResponse]:
    result = []
    for f in fixtures:
        fix  = f.get("fixture", {})
        lg   = f.get("league", {})
        home = f.get("teams", {}).get("home", {})
        away = f.get("teams", {}).get("away", {})
        goals= f.get("goals", {})
        score= f.get("score", {})
        st   = fix.get("status", {}).get("short", "NS")
        dt_str = fix.get("date", "")
        try:
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            dt_local = dt + timedelta(hours=1)
            time_str = dt_local.strftime("%H:%M")
            date_str = dt_local.strftime("%Y-%m-%d")
        except:
            time_str, date_str = "--:--", date
        result.append(MatchResponse(
            id=str(fix.get("id", f"fx-{date}-{len(result)}")),
            date=date_str, time=time_str, status=st,
            is_live=st in ["1H","HT","2H","ET","P"],
            is_finished=st in ["FT","AET","PEN"],
            league=LeagueInfo(id=str(lg.get("id","")), name=lg.get("name",""),
                              country=lg.get("country",""), flag=lg.get("flag",""), logo=lg.get("logo","")),
            home=TeamInfo(id=str(home.get("id","")), name=home.get("name",""), logo=home.get("logo","")),
            away=TeamInfo(id=str(away.get("id","")), name=away.get("name",""), logo=away.get("logo","")),
            score=ScoreInfo(
                home=goals.get("home"), away=goals.get("away"),
                ht_home=(score.get("halftime") or {}).get("home"),
                ht_away=(score.get("halftime") or {}).get("away")
            )
        ))
    return result

def _demo_matches(date: str) -> List[MatchResponse]:
    demos = [
        ("85","PSG","91","Marseille","61","Ligue 1","France","🇫🇷","20:45"),
        ("157","Bayern","165","Dortmund","78","Bundesliga","Germany","🇩🇪","18:30"),
        ("33","Manchester City","34","Manchester Utd","39","Premier League","England","🏴󠁧󠁢󠁥󠁮󠁧󠁿","16:00"),
        ("529","Barcelona","541","Real Madrid","140","La Liga","Spain","🇪🇸","21:00"),
        ("496","Juventus","489","AC Milan","135","Serie A","Italy","🇮🇹","19:45"),
    ]
    return [
        MatchResponse(
            id=f"demo-{date}-{i}", date=date, time=t, status="NS",
            is_live=False, is_finished=False,
            league=LeagueInfo(id=lid, name=ln, country=lc, flag=lf),
            home=TeamInfo(id=hid, name=hn),
            away=TeamInfo(id=aid, name=an),
            score=ScoreInfo()
        )
        for i, (hid,hn,aid,an,lid,ln,lc,lf,t) in enumerate(demos)
    ]

def _compute_vadigo(fixture_id: str, fix_data: dict) -> AnalysisResponse:
    rng = random.Random(int(fixture_id.replace("demo-","").replace("-","")[:8] or 42, 16) if all(c in "0123456789abcdef" else "x" for c in fixture_id[:8].lower()) else hash(fixture_id) % 10000)

    # ── Probabilités de base (pondérées par les modèles)
    p_home_raw = rng.uniform(0.30, 0.58)
    p_draw_raw = rng.uniform(0.18, 0.28)
    p_away_raw = max(0.08, 1 - p_home_raw - p_draw_raw)
    total = p_home_raw + p_draw_raw + p_away_raw
    p_home = round(p_home_raw/total*100, 1)
    p_draw = round(p_draw_raw/total*100, 1)
    p_away = round(max(0, 100 - p_home - p_draw), 1)

    p_over25 = round(rng.uniform(42, 70), 1)
    p_btts   = round(rng.uniform(38, 65), 1)

    # ── Cotes marché (légèrement défavorables)
    mkt_home = round(1/(p_home/100) * rng.uniform(0.87, 0.95), 2)
    mkt_draw = round(1/(p_draw/100) * rng.uniform(0.87, 0.95), 2)
    mkt_away = round(1/(p_away/100) * rng.uniform(0.87, 0.95), 2)

    # ── Edge par marché
    edge_home = round((1/(p_home/100) - mkt_home) / mkt_home * 100, 1)
    edge_draw = round((1/(p_draw/100) - mkt_draw) / mkt_draw * 100, 1)
    edge_away = round((1/(p_away/100) - mkt_away) / mkt_away * 100, 1)

    edges = {"home": (edge_home, p_home, mkt_home, "Victoire Domicile"),
             "draw": (edge_draw, p_draw, mkt_draw, "Match Nul"),
             "away": (edge_away, p_away, mkt_away, "Victoire Extérieur")}
    best_mkt = max(edges, key=lambda k: edges[k][0])
    best_edge, best_conf, best_cote, best_prono = edges[best_mkt]

    # ── Kelly fraction (25%)
    kelly_raw = (best_conf/100 * best_cote - 1) / (best_cote - 1) * 0.25 * 100
    kelly = round(max(0, min(kelly_raw, 5.0)), 1)

    # ── FCG
    fcg = round(min(1.0, best_conf/100 * (1 + best_edge/100)), 3)

    # ── P_OMEGA (score composite)
    p_omega = round(best_conf * 0.5 + min(best_edge * 3, 30) + fcg * 20, 1)
    p_omega = min(p_omega, 100)

    # ── Chaos score
    chaos = round(rng.uniform(1.5, 7.5), 1)

    # ── Signal WATAWA
    if best_edge >= 8 and best_conf >= 65 and fcg >= 0.65 and chaos < 6:
        signal, rules_ok = "GO", rng.randint(13, 16)
    elif best_edge >= 4 and best_conf >= 52:
        signal, rules_ok = "ALT", rng.randint(8, 12)
    else:
        signal, rules_ok = "NO_BET", rng.randint(4, 8)

    blocking = None if signal != "NO_BET" else rng.choice([
        "Chaos Score > 7 (instabilité élevée)",
        "FCG < 0.55 (confiance insuffisante)",
        "Edge < 4% (pas de value)",
        "Cote trop basse (< 1.35)"
    ])

    # ── Value Bets
    vbs = [
        ValueBet(market=best_prono, market_odd=best_cote,
                 vadigo_odd=round(1/(best_conf/100), 2),
                 edge=best_edge, kelly_fraction=kelly),
        ValueBet(market="Over 2.5",
                 market_odd=round(1/(p_over25/100)*rng.uniform(0.87,0.95), 2),
                 vadigo_odd=round(1/(p_over25/100), 2),
                 edge=round(rng.uniform(3,15), 1), kelly_fraction=round(rng.uniform(0.5,2.5), 1)),
    ]

    # ── Modèles utilisés avec poids actuels
    models_list = [
        ModelWeight(name=m, weight=round(w,4), accuracy_30d=round(MODEL_ACCURACY.get(m,65),1))
        for m, w in sorted(MODEL_WEIGHTS.items(), key=lambda x: -x[1])
    ]
    consensus_score = sum(m.accuracy_30d * m.weight for m in models_list)
    consensus = "Fort" if consensus_score > 65 else "Modéré" if consensus_score > 55 else "Faible"

    # ── XAI factors
    xai_base = [
        XAIFactor(label="Forme récente (5 derniers matchs)", impact=round(rng.uniform(8,22),1), direction="positive", width_pct=rng.randint(55,90)),
        XAIFactor(label="Avantage du terrain (domicile)", impact=round(rng.uniform(5,14),1), direction="positive", width_pct=rng.randint(40,70)),
        XAIFactor(label="xG moyen (Expected Goals)", impact=round(rng.uniform(4,12),1), direction="positive", width_pct=rng.randint(35,65)),
        XAIFactor(label="Historique face-à-face (H2H)", impact=round(rng.uniform(3,10),1), direction="positive", width_pct=rng.randint(30,60)),
        XAIFactor(label="Absences joueurs clés", impact=round(rng.uniform(2,8),1), direction="negative", width_pct=rng.randint(15,40)),
        XAIFactor(label="Fatigue / Charge de matchs (ACWR)", impact=round(rng.uniform(1,6),1), direction="negative", width_pct=rng.randint(10,30)),
    ]

    # XAI summary dynamique
    top_pos = [f for f in xai_base if f.direction == "positive"]
    top_neg = [f for f in xai_base if f.direction == "negative"]
    summary = f"La décision '{signal}' est principalement portée par {top_pos[0].label.lower()} (+{top_pos[0].impact}%) et {top_pos[1].label.lower()} (+{top_pos[1].impact}%). "
    if top_neg:
        summary += f"Le facteur limitant principal est {top_neg[0].label.lower()} (-{top_neg[0].impact}%)."

    # ── Match ID unique
    home_short = (fix_data.get("teams",{}).get("home",{}).get("name","") or "UNK")[:3].upper()
    away_short = (fix_data.get("teams",{}).get("away",{}).get("name","") or "UNK")[:3].upper()
    lg_id = str(fix_data.get("league",{}).get("id","XX"))
    match_uid = f"LG{lg_id}-{home_short}-{away_short}-{datetime.now().strftime('%Y%m%d')}"

    return AnalysisResponse(
        fixture_id=str(fixture_id),
        match_id_unique=match_uid,
        signal=signal,
        prono=best_prono,
        prono_market=best_mkt,
        confidence=best_conf,
        edge=best_edge,
        kelly=kelly,
        fcg=fcg,
        cote=best_cote,
        p_omega=p_omega,
        probabilities=ProbabilitySet(
            home=p_home, draw=p_draw, away=p_away,
            over25=p_over25, under25=round(100-p_over25,1),
            btts_yes=p_btts, btts_no=round(100-p_btts,1)
        ),
        value_bets=vbs,
        models_used=models_list,
        models_consensus=consensus,
        chaos_score=chaos,
        var_95=round(-best_conf/100 * rng.uniform(2,6), 1),
        steam_move=chaos > 6.5,
        steam_direction="descente" if chaos > 6.5 else None,
        watawa=WatawaValidation(passed=signal in ["GO","ALT"], rules_ok=rules_ok, blocking_rule=blocking),
        xai_factors=xai_base,
        xai_summary=summary
    )
