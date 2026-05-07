import os
import datetime
import jwt

from fastapi import (
    APIRouter,
    HTTPException,
    Depends
)

from fastapi.security import (
    HTTPBearer,
    HTTPAuthorizationCredentials
)

from pydantic import (
    BaseModel,
    EmailStr
)

from supabase import create_client

from modules.database import (
    get_connection
)

# =========================================================
# ROUTER
# =========================================================

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)

security = HTTPBearer()

SECRET_KEY = os.environ.get(
    "SECRET_KEY",
    "your-secret-key"
)

SUPABASE_URL = os.environ.get(
    "SUPABASE_URL"
)

SUPABASE_KEY = os.environ.get(
    "SUPABASE_KEY"
)

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

# =========================================================
# MODELS
# =========================================================

class SignUpRequest(BaseModel):
    name: str
    address: str
    email: EmailStr
    mobile_number: str
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    name: str
    address: str
    email: EmailStr
    mobile_number: str
    token: str


# =========================================================
# JWT
# =========================================================

def create_token(email: str):

    payload = {
        "sub": email,
        "exp": datetime.datetime.utcnow()
        + datetime.timedelta(days=7)
    }

    return jwt.encode(
        payload,
        SECRET_KEY,
        algorithm="HS256"
    )


def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):

    try:

        payload = jwt.decode(
            credentials.credentials,
            SECRET_KEY,
            algorithms=["HS256"]
        )

        return payload["sub"]

    except jwt.ExpiredSignatureError:

        raise HTTPException(
            status_code=401,
            detail="Token expired"
        )

    except jwt.InvalidTokenError:

        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )


# =========================================================
# DATABASE HELPERS
# =========================================================

def get_profile_by_email(email: str):

    conn = get_connection()
    cur = conn.cursor()

    try:

        cur.execute("""
            SELECT
                id,
                name,
                address,
                email,
                mobile_number
            FROM users
            WHERE email = %s
        """, (email,))

        row = cur.fetchone()

        if not row:
            return None

        return {
            "id": str(row[0]),
            "name": row[1],
            "address": row[2],
            "email": row[3],
            "mobile_number": row[4],
        }

    finally:
        cur.close()
        conn.close()


def create_profile(
    name: str,
    address: str,
    email: str,
    mobile_number: str
):

    conn = get_connection()
    cur = conn.cursor()

    try:

        cur.execute("""
            INSERT INTO users (
                name,
                address,
                email,
                mobile_number
            )
            VALUES (%s, %s, %s, %s)
        """, (
            name,
            address,
            email,
            mobile_number
        ))

        conn.commit()

    finally:
        cur.close()
        conn.close()


# =========================================================
# SIGNUP
# =========================================================

@router.post(
    "/signup",
    response_model=UserResponse
)
def signup(data: SignUpRequest):

    existing_user = get_profile_by_email(
        data.email
    )

    if existing_user:

        raise HTTPException(
            status_code=400,
            detail="Email already exists"
        )

    # =====================================================
    # CREATE AUTH ACCOUNT
    # =====================================================

    auth_response = supabase.auth.sign_up({
        "email": data.email,
        "password": data.password
    })

    if not auth_response.user:

        raise HTTPException(
            status_code=400,
            detail="Signup failed"
        )

    # =====================================================
    # CREATE PROFILE
    # =====================================================

    create_profile(
        data.name,
        data.address,
        data.email,
        data.mobile_number
    )

    token = create_token(
        data.email
    )

    return UserResponse(
        name=data.name,
        address=data.address,
        email=data.email,
        mobile_number=data.mobile_number,
        token=token
    )


# =========================================================
# LOGIN
# =========================================================

@router.post(
    "/login",
    response_model=UserResponse
)
def login(data: LoginRequest):

    auth_response = supabase.auth.sign_in_with_password({
        "email": data.email,
        "password": data.password
    })

    if not auth_response.user:

        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    user = get_profile_by_email(
        data.email
    )

    if not user:

        raise HTTPException(
            status_code=404,
            detail="Profile not found"
        )

    token = create_token(
        data.email
    )

    return UserResponse(
        name=user["name"],
        address=user["address"],
        email=user["email"],
        mobile_number=user["mobile_number"],
        token=token
    )


# =========================================================
# CURRENT USER
# =========================================================

@router.get("/me")
def get_me(
    email: str = Depends(verify_token)
):

    user = get_profile_by_email(
        email
    )

    if not user:

        raise HTTPException(
            status_code=404,
            detail="User not found"
        )

    return user