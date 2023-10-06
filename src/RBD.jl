module RBD

export
    SolverBody, SolverConstraint3DofLinear, SolverConstraint1Dof, SolverConstraint1DofAngular, SolverConstraintRigid,
    solve, projectToManifoldVelocity, projectToR3Velocity!,
    setupPointToPointConstraint, setup1DofConstraint, setup1DofAngularConstraint, setupRigidConstraint,
    skew, translation, create_transform, transform, inertia_tensor, rotation, combine_inertia,
    Matrix3x3, Vector3, Matrix4x4, Transform, step_world, RigidBody, to_solver_body,
    RigidBodyConstraint, PointToPointConstraint, Linear1DofConstraint, ContactConstraint, ContactManifoldConstraint, Angular1DofConstraint, RigidConstraint,
    ManifoldContact,
    ndof, bodyA, bodyB, body_ids,
    solve_sequential_impulse,
    to_solver_constraint, linearJa, linearJb, angularJa, angularJb, rhs,
    global_jacobian, global_mass_inv,
    solve_global, global_system_matrix, forcepsd,
    reduce_LDLt, solve_global_block, generate_constraint_graph, cross_term, diagonal_term, LDLt_to_Arrays,
    assemble_block_stream, precompute_block_solver,
    BlockArray, BlockVector,
    choln, blockrange, parse_log, LBlock_to_L, cholupdate, choldowndate,
    remove_constraint, remove_LLt


using LinearAlgebra, StaticArrays, ReferenceFrameRotations, SparseArrays

#===== BlockArray ========#

struct BlockArray{N} <: AbstractArray{Float64,2}
    A::AbstractArray
end

struct BlockVector{N} <: AbstractArray{Float64,1}
    A::AbstractArray
end

blockrange(i,j,n) = CartesianIndex((i[1]-1)*n+1, (j[1]-1)*n+1):CartesianIndex(i[end]*n,j[end]*n)
blockrange(i,n) = CartesianIndex((i[1]-1)*n+1, 1):CartesianIndex(i[end]*n,n)


Base.getindex(A::T, i::R, j::R2) where {R<:AbstractRange, R2<:Integer, N, T <: BlockArray{N}} = BlockVector{N}(A.A[blockrange(i,j,N)])
Base.getindex(A::T, i::Colon, j::R2) where {R2<:Integer, N, T <: BlockArray{N}} = BlockVector{N}(A.A[blockrange(1:size(A)[1],j,N)])
Base.getindex(A::T, i::R, j::R) where {R, N, T <: BlockArray{N}} = BlockArray{N}(A.A[blockrange(i,j,N)])
Base.getindex(A::T, i::BitVector, j::BitVector) where {N, T <: BlockArray{N}} = BlockArray{N}(A.A[
                                                                                                mapreduce(b->repeat([b], N), vcat, i),
                                                                                                mapreduce(b->repeat([b], N), vcat, j)])

#Base.getindex(A::T, i, j) where {N, T <: BlockArray{N}} = A.A[blockrange(i,j,N)]
Base.setindex!(A::T, a::U, i, j) where {N, U <: BlockArray{N}, T <: BlockArray{N}} = A.A[blockrange(i,j,N)] = a.A
Base.setindex!(A::T, a::U, i, j) where {N, U <: BlockVector{N}, T <: BlockArray{N}} = A.A[blockrange(i,j,N)] = a.A
Base.setindex!(A::T, a::U, i, j) where {N, U <: AbstractArray, T <: BlockArray{N}} = A.A[blockrange(i,j,N)] = a
Base.getindex(A::T, i::R) where {N, R<:Integer, T <: BlockVector{N}} = BlockArray{N}(A.A[blockrange(i,N)])
Base.getindex(A::T, i) where {N, T <: BlockVector{N}} = BlockVector{N}(A.A[blockrange(i,N)])
Base.setindex!(A::T, a::U, i) where {N, U <: BlockArray{N}, T <: BlockVector{N}} = A.A[blockrange(i,N)] = a.A
Base.setindex!(A::T, a::U, i) where {N, U <: BlockVector{N}, T <: BlockVector{N}} = A.A[blockrange(i,N)] = a.A
Base.setindex!(A::T, a::U, i) where {N, U <: AbstractArray, T <: BlockVector{N}} = A.A[blockrange(i,N)] = a
Base.length(x::T) where {N, T<:BlockVector{N}} = div(size(x.A)[1], N)
Base.axes(A::BlockArray{N}, d) where N = Base.OneTo(div(size(A.A)[d], N))

Base.:+(A::T, B::U) where {N,T<:BlockArray{N}, U<:BlockArray{N}} = BlockArray{N}(A.A + B.A)
Base.:+(A::T, B::U) where {N,T<:BlockVector{N}, U<:BlockVector{N}} = BlockVector{N}(A.A + B.A)
Base.:-(A::T, B::U) where {N,T<:BlockArray{N}, U<:BlockArray{N}} = BlockArray{N}(A.A - B.A)
Base.:-(A::T, B::U) where {N,T<:BlockVector{N}, U<:BlockVector{N}} = BlockVector{N}(A.A - B.A)
Base.:*(A::T, B::U) where {N,T<:BlockArray{N}, U<:BlockArray{N}} = BlockArray{N}(A.A * B.A)
Base.zero(A::BlockArray{N}) where N = BlockArray{N}(zero(A.A))
LinearAlgebra.det(A::BlockArray{N}) where N = det(A.A)

function Base.:*(A::T, B::U) where {N,T<:BlockArray{N},U<:BlockVector{N}}
    C = deepcopy(B)
    for i in 1:length(B)
        C[i] = A * B[i]
    end
    C
end

function Base.:*(A::T, B::U) where {N,T<:BlockVector{N},U<:BlockArray{N}}
    C = deepcopy(A)
    for i in 1:length(A)
        C[i] = A[i] * B
    end
    C
end

function Base.:/(A::T, B::U) where {N,T<:BlockVector{N},U<:BlockArray{N}}
    C = deepcopy(A)
    for i in 1:length(B)
        C[i] = C[i] / B
    end
    C
end

Base.:/(A::T, B::U) where {N,T<:BlockArray{N}, U<:BlockArray{N}} = BlockArray{N}(A.A / B.A)
LinearAlgebra.inv(A::BlockArray{N}) where N = BlockArray{N}(inv(A.A))
Base.:sqrt(A::T) where {N,T<:BlockArray{N}} = BlockArray{N}(choln(A.A))
LinearAlgebra.adjoint(A::T) where{N, T<:BlockArray{N}} = BlockArray{N}(adjoint(A.A))
Base.size(A::T) where {N,T<:BlockArray{N}} = map(x->div(x, N), size(A.A))
Base.size(A::T) where {N,T<:BlockVector{N}} = map(x->div(x, N), size(A.A))
Base.display(A::BlockArray{N}) where N = display((typeof(A), A.A))
Base.show(::IO, A::BlockArray{N}) where N = println( typeof(A), A.A )
Base.display(A::BlockVector{N}) where N = display((typeof(A), A.A))
Base.show(::IO, A::BlockVector{N}) where N = println( typeof(A), A.A )


#===== end BlockArray ========#

Scalar = Float64
Vector3 = SVector{3,Scalar}
Vector6 = SVector{6,Scalar}
Matrix3x3 = SMatrix{3,3,Scalar,9}
Matrix4x4 = SMatrix{4,4,Scalar,16}
Matrix6x6 = SMatrix{6,6,Scalar,36}
Matrix3x6 = SMatrix{3,6,Scalar,18}
Matrix6x3 = SMatrix{6,3,Scalar,18}
Transform = Matrix4x4

transform(A::AT, v::VT) where {T,S,AT<:SMatrix{4,4,T,16},VT<:SVector{3,S}} = SVector{3,S}((A*vcat(v, 1))[1:3])
transform(A::AT, v::VT) where {T,S,AT<:Matrix{T},VT<:SVector{3,S}} = SVector{3,S}((A*vcat(v, 1))[1:3])
rotate(A::AT, v) where {T,AT<:SMatrix{4,4,T,16}} = SVector{3,T}(A[1:3, 1:3] * v)
rotate(A::AT, v) where {T,AT<:Matrix{T}} = SVector{3,T}(A[1:3, 1:3] * v)
translation(wTb) = Vector3(wTb[4, 1:3])
rotation(A::Transform) = Matrix3x3(A[1:3, 1:3])

create_transform(R, t) = vcat(hcat(R, zeros(3)), hcat(t', 1)) |> Transform

Base.:>>>(q::T, v) where {T<:Quaternion} = vect(q * v * inv(q))

skew(ω::Vector3) = [0.0 -ω[3] ω[2]
    ω[3] 0.0 -ω[1]
    -ω[2] ω[1] 0.0]

function inertia_tensor(box::Vector3, m::Scalar)
    x, y, z = box
    (m / 12.0) * Vector3(y^2 + z^2, x^2 + z^2, x^2 + y^2)
end

# combine inertia tensor into the reference frame of a.  often a will be the world frame
function combine_inertia(inertiaa::Matrix3x3,  inertiab::Matrix3x3, massb::Scalar, aTb::Transform)
    d = translation(aTb)
    inertiaa + inertiab + massb*(diagm(d⋅d) - d*d')
end

function combine_inertia(inertiaa::Matrix3x3, wTa::Transform, inertiab::Vector3, massb::Scalar, mass_frame::Transform, wTb::Transform)
end

function generate_frame(z)
    y = Vector3(0, 1, 0)
    if abs(y ⋅ z) < 0.9
        x = normalize(y × z)
        y = z × x
    else
        x = Vector3(1, 0, 0)
        y = normalize(z × x)
        x = y × z
    end
    (x, y, z)
end

import LinearAlgebra.normalize

normalize(q::Quaternion) = q / norm(q)

angle_axis(v::T, θ) where {T<:AbstractVector} = I + sin(θ) * skew(v) + (1 - cos(θ)) * skew(v) * skew(v)

## cholesky factorization  L*L' = A
## want to find Lij ( could be subblocks ) where
## [ L11 0   0   .. ] [ L11' L21' L31' .. ]   [ A11 A12 A13 .. ]
## [ L21 L22 0   .. ] [ 0    L22' L32' .. ]   [ A21 A22 A23 .. ]
## [ L31 L32 L33 .. ] [ 0    0    L33' .. ] = [ A31 A32 A33 .. ]
## [ ..  ..  ..  .. ] [ ..   ..   ..   .. ]   [ ..  ..  ..  .. ]
## 
## Aij = Aji', i.e. A12 = A21'
##
## Basic relation is Lij = (Aij - SUM_k=1:j-1( Lik*Ljk' ))*inv(Ljj')
## When i == j then Lii = sqrt(Aii - SUM_k=1:i-1( Lik*Lik' ))
## Examples
## L11 = sqrt( A11 );   L21 = A21*inv(L11');   L32 = (A32 - L31*L21')*inv(L22)'
##
## sqrt(subblock) should be cholesky factorization
function choln(A)
    n = first(size(A))
    lower = zero(A)
    matrix = A

    for i in 1:n
        for j in 1:i
            sum1 = zero(matrix[1,1]);

            for k in 1:j
                sum1 += lower[i,k] * lower[j,k]';
            end

            if (j == i)
                lower[j,j] = sqrt(matrix[j,j] - sum1)
            else
                # Evaluating L(i, j)  using L(j, j)
                if(det(lower[j,j]) > 0)
                    lower[i,j] = (matrix[i,j] - sum1)*inv(lower[j,j]')
                else
                    throw("det(lower[j,j]) < 0")
                end#if
            end#if else
        end#for
    end#for
    return lower
end

##=========== Constraint Solver ==============##


#== Solver types ==#

mutable struct SolverBody
    v::Vector3
    ω::Vector3

    #Iinv::Vector3   # hmmm... if we want in localspace need transform stored
    #priciple_axis_transform::Quaternion   # transform into space where inertia tensor I is diagonal
    wTb::Transform

    Iinv::Matrix3x3   # in worldspace
    massInv::Scalar

    id::Int64
end

abstract type SolverConstraint end
abstract type SolverConstraintClamped <: SolverConstraint end

linearJa(c::T) where {T<:SolverConstraint} = zeros(3)
linearJb(c::T) where {T<:SolverConstraint} = zeros(3)
angularJa(c::T) where {T<:SolverConstraint} = zeros(3)
angularJb(c::T) where {T<:SolverConstraint} = zeros(3)
ndof(c::T) where {T<:SolverConstraint} = 1
bodyA(c::T) where {T<:SolverConstraint} = c.bodyA
bodyB(c::T) where {T<:SolverConstraint} = c.bodyB
rhs(c::T) where {T<:SolverConstraint} = c.manifoldVelocityToCorrect + projectToManifoldVelocity(c)

mutable struct SolverConstraint1Dof <: SolverConstraint

    manifoldVelocityToCorrect::Scalar
    solutionImpulse::Scalar

    bodyA::SolverBody
    bodyB::SolverBody

    Ja::Vector3  # Jacobian for body A
    Jb::Vector3  # Jacobian for body B

    axis::Vector3  # world space axis of constraint

    Dinv::Scalar # effective mass inverse

end

SolverConstraint1Dof(bodyA, bodyB) = SolverConstraint1Dof(0, 0, bodyA, bodyB, Vector3(zeros(3)), Vector3(zeros(3)), Vector3(zeros(3)), 0)

linearJa(c::T) where {T<:SolverConstraint1Dof} = c.axis
linearJb(c::T) where {T<:SolverConstraint1Dof} = -c.axis
angularJa(c::T) where {T<:SolverConstraint1Dof} = c.Ja
angularJb(c::T) where {T<:SolverConstraint1Dof} = c.Jb

mutable struct SolverConstraint1DofAngular <: SolverConstraint

    manifoldVelocityToCorrect::Scalar
    solutionImpulse::Scalar

    bodyA::SolverBody
    bodyB::SolverBody

    Ja::Vector3  # Jacobian for body A
    Jb::Vector3  # Jacobian for body B

    Dinv::Scalar # effective mass inverse

end

SolverConstraint1DofAngular(bodyA, bodyB) = SolverConstraint1DofAngular(0, 0, bodyA, bodyB, Vector3(zeros(3)), Vector3(zeros(3)), 0)

linearJa(c::T) where {T<:SolverConstraint1DofAngular} = zeros(3)
linearJb(c::T) where {T<:SolverConstraint1DofAngular} = zeros(3)
angularJa(c::T) where {T<:SolverConstraint1DofAngular} = c.Ja
angularJb(c::T) where {T<:SolverConstraint1DofAngular} = c.Jb

mutable struct SolverConstraintContact <: SolverConstraintClamped
    normal_constraint::SolverConstraint1Dof
    friction_constraint_id::Int64  #!me I think I'm reversing this and having all the contact constraints contained by the manifold, in that case, this isn't used
end

SolverConstraintContact(bodyA::T, bodyB::T) where {T<:SolverBody} = SolverConstraintContact(
    SolverConstraint1Dof(0, 0, bodyA, bodyB, Vector3(zeros(3)), Vector3(zeros(3)), Vector3(zeros(3)), 0), 0)

unclamped_constraint(c::SolverConstraintContact) = c.normal_constraint

struct ManifoldContact
    ra::Vector3  # offset from com to pivot in space of body A
    rb::Vector3
end

struct SolverConstraintContactManifoldPoint

    manifoldVelocityToCorrect::Scalar
    solutionImpulse::Scalar

    Ja::Vector3  # Jacobian for body A
    Jb::Vector3  # Jacobian for body B

    Dinv::Scalar # effective mass inverse

end

# n 1dof contact constraint + 2 tangential friction constraints + 1 angular( twist ) friction constraint
mutable struct SolverConstraintContactManifold

    manifoldVelocityToCorrect::Vector3
    solutionImpulse::Vector3

    bodyA::SolverBody
    bodyB::SolverBody

    linearJa::Matrix3x3  # column 1 twist
    linearJb::Matrix3x3

    angularJa::Matrix3x3
    angularJb::Matrix3x3

    Dinv::Matrix3x3 # effective mass inverse

    mu::Scalar

    normalImpulse::Scalar
    contactPoints::Vector{SolverConstraintContact}
    #contactPoints::Vector{SolverConstraintContactPoint}
end

SolverConstraintContactManifold(bodyA, bodyB) = SolverConstraintContactManifold(zero(Vector3), zero(Vector3), bodyA, bodyB, zero(Matrix3x3), zero(Matrix3x3), zero(Matrix3x3), zero(Matrix3x3), zero(Matrix3x3), 0.7, 0, [])

mutable struct SolverConstraint3DofLinear <: SolverConstraint

    manifoldVelocityToCorrect::Vector3
    solutionImpulse::Vector3

    bodyA::SolverBody
    bodyB::SolverBody

    angularJa::Matrix3x3  # Jacobian for body A
    angularJb::Matrix3x3  # Jacobian for body B

    Dinv::Matrix3x3 # effective mass inverse

end

SolverConstraint3DofLinear(bodyA, bodyB) = SolverConstraint3DofLinear(
    Vector3(zeros(3)), Vector3(zeros(3)),
    bodyA, bodyB,
    Matrix3x3(zeros(3, 3)), Matrix3x3(zeros(3, 3)), Matrix3x3(zeros(3, 3)))

ndof(c::SolverConstraint3DofLinear) = 3
linearJa(c::T) where {T<:SolverConstraint3DofLinear} = Matrix3x3(diagm(ones(3)))
linearJb(c::T) where {T<:SolverConstraint3DofLinear} = -Matrix3x3(diagm(ones(3)))
angularJa(c::T) where {T<:SolverConstraint3DofLinear} = c.angularJa
angularJb(c::T) where {T<:SolverConstraint3DofLinear} = c.angularJb

mutable struct SolverConstraintRigid <: SolverConstraint

    manifoldVelocityToCorrect::Vector6
    solutionImpulse::Vector6

    bodyA::SolverBody
    bodyB::SolverBody

    angularJa::Matrix3x3  # Jacobian for body A
    angularJb::Matrix3x3  # Jacobian for body B

    frame::Matrix3x3

    Dinv::Matrix6x6 # effective mass inverse

end

SolverConstraintRigid(bodyA, bodyB) = SolverConstraintRigid(
    Vector6(zeros(6)), Vector6(zeros(6)),
    bodyA, bodyB,
    zero(Matrix3x3), zero(Matrix3x3), zero(Matrix3x3), zero(Matrix6x6))

ndof(c::SolverConstraintRigid) = 6
linearJa(c::T) where {T<:SolverConstraintRigid} = Matrix6x3(vcat(diagm(ones(3)), zeros(3, 3)))
linearJb(c::T) where {T<:SolverConstraintRigid} = Matrix6x3(vcat(-diagm(ones(3)), zeros(3, 3)))
angularJa(c::T) where {T<:SolverConstraintRigid} = Matrix6x3(vcat(c.angularJa, c.frame'))
angularJb(c::T) where {T<:SolverConstraintRigid} = Matrix6x3(vcat(c.angularJb, -c.frame'))

#== end Solver types ==#


function setup1DofConstraint(dt_inv, wTa::Transform, wTb::Transform, pointOnAInA::Vector3, pointOnBInB::Vector3, worldAxis::Vector3, c::SolverConstraint1Dof)
    r1 = rotate(wTa, pointOnAInA)
    r2 = rotate(wTb, pointOnBInB)

    c.Ja = r1 × worldAxis  # non-permissable angular axis
    c.Jb = -r2 × worldAxis

    c.Dinv = inv((c.bodyA.Iinv * c.Ja) ⋅ c.Ja + (c.bodyB.Iinv * c.Jb) ⋅ c.Jb + c.bodyA.massInv + c.bodyB.massInv)
    c.axis = worldAxis

    x1 = translation(wTa) + r1
    x2 = translation(wTb) + r2

    # Compute the positional constraint error (scaled by the Baumgarte coefficient 'beta')
    positionError = (x1 - x2)
    #!me scaleError = dt_inv * 0.2
    scaleError = dt_inv * 1.0  #!me I think we want to work with actual position error at this point.  do same for others?  or maybe just use for contacts?
    positionError = positionError * scaleError

    c.manifoldVelocityToCorrect = positionError ⋅ c.axis

    return c
end


function projectToManifoldVelocity(c::SolverConstraint1Dof)
    p = c.bodyA.v ⋅ c.axis
    p -= c.bodyB.v ⋅ c.axis

    p += c.Ja ⋅ c.bodyA.ω
    p += c.Jb ⋅ c.bodyB.ω

    return p
end

function projectToR3Velocity!(c::SolverConstraint1Dof, dp::Scalar)
    dv = dp * c.bodyA.massInv
    c.bodyA.v += dv * c.axis
    dv = dp * c.bodyB.massInv
    c.bodyB.v -= dv * c.axis
    dω = c.Ja * dp
    dω = c.bodyA.Iinv * dω
    c.bodyA.ω += dω
    dω = c.Jb * dp
    dω = c.bodyB.Iinv * dω
    c.bodyB.ω += dω
    c
end

"""
    provide two mutually perpendicular axis which are also perpendicular to the axis of the prohibited rotation.  One on body A and one on body B.
    this relation should be met when the constraint is at rest a⟂×b⟂ = constrained
"""
function setup1DofAngularConstraint(dt_inv, perp_axisa::Vector3, perp_axisb::Vector3, c::SolverConstraint1DofAngular)

    axis = normalize(perp_axisa × perp_axisb)
    c.Ja = axis
    c.Jb = -c.Ja

    c.Dinv = inv((c.bodyA.Iinv * c.Ja) ⋅ c.Ja + (c.bodyB.Iinv * c.Jb) ⋅ c.Jb)

    # Compute the constraint error (scaled by the Baumgarte coefficient 'beta')
    angularError = perp_axisa ⋅ perp_axisb
    scaleError = dt_inv * 0.2
    angularError = angularError * scaleError

    c.manifoldVelocityToCorrect = angularError

    return c
end


function projectToManifoldVelocity(c::SolverConstraint1DofAngular)
    p = c.Ja ⋅ c.bodyA.ω
    p += c.Jb ⋅ c.bodyB.ω

    return p
end

function projectToR3Velocity!(c::SolverConstraint1DofAngular, dp::Scalar)
    dω = c.Ja * dp
    dω = c.bodyA.Iinv * dω
    c.bodyA.ω += dω
    dω = c.Jb * dp
    dω = c.bodyB.Iinv * dω
    c.bodyB.ω += dω
    c
end

#== Contact ==#

function setupContactConstraint(dt_inv, wTa::Transform, wTb::Transform, pointOnAInA::Vector3, pointOnBInB::Vector3, worldAxis::Vector3, c::SolverConstraintContact)
    c.normal_constraint = setup1DofConstraint(dt_inv, wTa, wTb, pointOnAInA, pointOnBInB, worldAxis, c.normal_constraint)
    c
end

projectToManifoldVelocity(c::SolverConstraintContact) = projectToManifoldVelocity(c.normal_constraint)

function projectToR3Velocity!(c::SolverConstraintContact, dp::Scalar)
    c.normal_constraint = projectToR3Velocity!(c.normal_constraint, dp)
    c
end

#== Point to Point ===#

function projectToManifoldVelocity(c::SolverConstraint3DofLinear)
    p = c.bodyA.v
    p -= c.bodyB.v

    ppart = c.bodyA.ω
    p += c.angularJa * ppart

    ppart = c.bodyB.ω
    ppart = c.angularJb * ppart
    p += ppart

    return p
end

function projectToR3Velocity!(c::SolverConstraint3DofLinear, dp::Vector3)
    dv = dp * c.bodyA.massInv
    c.bodyA.v += dv
    dv = dp * c.bodyB.massInv
    c.bodyB.v -= dv
    dω = dp
    dω = c.angularJa' * dω
    dω = c.bodyA.Iinv * dω
    c.bodyA.ω += dω
    dω = dp
    dω = c.angularJb' * dω
    dω = c.bodyB.Iinv * dω
    c.bodyB.ω += dω
    c
end

projectToR3Velocity!(c::SolverConstraint3DofLinear, dp::V) where {V<:AbstractVector} = projectToR3Velocity!(c, Vector3(dp))

# output Dinv and solutionImpulse->zero
# to be more Julian should change to method returning Dinv
function init(c::SolverConstraint3DofLinear)
    Dinv = Matrix3x3(diagm(repeat([c.bodyA.massInv + c.bodyB.massInv], 3)))

    Jt = c.angularJa'
    dpartM = c.angularJa * c.bodyA.Iinv
    dpartM *= Jt
    Dinv += dpartM

    Jt = c.angularJb'
    dpartM = c.angularJb * c.bodyB.Iinv
    dpartM *= Jt
    Dinv += dpartM

    c.Dinv = Matrix3x3(inv(Dinv))

    c.solutionImpulse = zero(c.solutionImpulse)

    return c
end


function setupPointToPointConstraint(dt_inv, wTa::Transform, wTb::Transform, pointOnAInA::Vector3, pointOnBInB::Vector3, c)
    r1 = rotate(wTa, pointOnAInA)
    r2 = rotate(wTb, pointOnBInB)

    x1 = translation(wTa) + r1
    x2 = translation(wTb) + r2

    r1 = -r1
    c.angularJa = skew(r1)
    c.angularJb = skew(r2)

    # Compute the positional constraint error (scaled by the Baumgarte coefficient 'beta')
    positionError = (x1 - x2)
    scaleError = dt_inv * 0.2
    positionError = positionError * scaleError

    c.manifoldVelocityToCorrect = positionError

    return init(c)
end

#== 6DOF all (rigid) fixed ===#

function projectToManifoldVelocity(c::SolverConstraintRigid)
    p = c.bodyA.v
    p -= c.bodyB.v

    p += c.angularJa * c.bodyA.ω
    p += c.angularJb * c.bodyB.ω

    pω = c.frame' * c.bodyA.ω
    pω -= c.frame' * c.bodyB.ω

    return Vector6(p..., pω...)
end

function projectToR3Velocity!(c::SolverConstraintRigid, dj::Vector6)

    # linear dof part
    dp = Vector3(dj[1:3])
    dpω = Vector3(dj[4:6])
    dv = dp * c.bodyA.massInv
    c.bodyA.v += dv
    dv = dp * c.bodyB.massInv
    c.bodyB.v -= dv
    dω = c.angularJa' * dp
    dω = c.bodyA.Iinv * dω
    c.bodyA.ω += dω
    dω = c.angularJb' * dp
    dω = c.bodyB.Iinv * dω
    c.bodyB.ω += dω

    # angular dof part
    dω = c.frame * dpω
    dω = c.bodyA.Iinv * dω
    c.bodyA.ω += dω
    dω = -c.frame * dpω
    dω = c.bodyB.Iinv * dω
    c.bodyB.ω += dω

    c
end

projectToR3Velocity!(c::SolverConstraintRigid, dp::V) where {V<:AbstractVector} = projectToR3Velocity!(c, Vector6(dp))


# output Dinv and solutionImpulse->zero
# to be more Julian should change to method returning Dinv
function diagonal_term(c::SolverConstraintRigid)
    D11 = Matrix3x3(diagm(repeat([c.bodyA.massInv + c.bodyB.massInv], 3)))

    F = c.frame

    Jat = c.angularJa'
    JaI = c.angularJa * c.bodyA.Iinv
    D11 += JaI * Jat

    Jbt = c.angularJb'
    JbI = c.angularJb * c.bodyB.Iinv
    D11 += JbI * Jbt

    D12 = JaI * F - JbI * F

    D21 = F' * JaI' - F' * JbI'  # inertia tensor is symmetric so (J*I)' = I*J' which goes in D21

    D22 = F' * c.bodyA.Iinv * F + F' * c.bodyB.Iinv * F

    D = [hcat(D11, D12)
        hcat(D21, D22)]
end

function init(c::SolverConstraintRigid)
    D = diagonal_term(c)
    if det(D) != 0
        c.Dinv = Matrix6x6(inv(D))
    end

    c.solutionImpulse = zero(c.solutionImpulse)

    return c
end


function cross_term(c1::SolverConstraintRigid, c2::SolverConstraintRigid)

    # figure out which body is shared, this is the lumped mass matrix doing the mixing
    F1 = c1.frame
    F2 = c2.frame

    # note mass changes sign depending on how a and b mixes
    # frames on b have -ve sign
    if c1.bodyA == c2.bodyA
        minv = c1.bodyA.massInv
        Iinv = c1.bodyA.Iinv
        J1 = c1.angularJa
        J2 = c2.angularJa
    elseif c1.bodyA == c2.bodyB
        minv = -c1.bodyA.massInv
        Iinv = c1.bodyA.Iinv
        J1 = c1.angularJa
        J2 = c2.angularJb
        F2 = -F2
    elseif c1.bodyB == c2.bodyA
        minv = -c1.bodyB.massInv
        Iinv = c1.bodyB.Iinv
        J1 = c1.angularJb
        J2 = c2.angularJa
        F1 = -F1
    else
        minv = c1.bodyB.massInv
        Iinv = c1.bodyB.Iinv
        J1 = c1.angularJb
        J2 = c2.angularJb
        F1 = -F1
        F2 = -F2
    end

    D11 = [minv 0.0 0.0
        0.0 minv 0.0
        0.0 0.0 minv]

    J1I = J1 * Iinv
    J2I = J2 * Iinv
    D11 = D11 + J1I * J2'
    D12 = J1I * F2
    D21 = F1' * J2I' # inertia tensor is symmetric so (J*I)' = I*J' which goes in D21
    D22 = F1' * Iinv * F2

    [D11 D12
        D21 D22]
end


function setupRigidConstraint(dt_inv, wTa::Transform, wTb::Transform, pointOnAInA::Vector3, pointOnBInB::Vector3, Fa::Matrix3x3, Fb::Matrix3x3, c)
    r1 = rotate(wTa, pointOnAInA)
    r2 = rotate(wTb, pointOnBInB)

    x1 = translation(wTa) + r1
    x2 = translation(wTb) + r2

    r1 = -r1
    c.angularJa = skew(r1)
    c.angularJb = skew(r2)

    # Compute the positional constraint error (scaled by the Baumgarte coefficient 'beta')
    positionError = (x1 - x2)
    scaleError = dt_inv * 0.2
    positionError = positionError * scaleError

    # angular part
    xa, ya, za = eachcol(Fa)
    xb, yb, zb = eachcol(Fb)
    angularError = Vector3(ya ⋅ zb, za ⋅ xb, xa ⋅ yb)
    scaleError = dt_inv * 0.2
    angularError = angularError * scaleError

    c.manifoldVelocityToCorrect = Vector6(positionError..., angularError...)

    xw = normalize(ya × zb)
    yw = normalize(za × xw)
    c.frame = Matrix3x3(hcat(xw, yw, xw × yw))

    return init(c)
end


function setupRigidConstraint(dt, rba::SolverBody, rbb::SolverBody)
    c = SolverConstraintRigid(rba, rbb)
    anchor = (translation(rba.wTb) + translation(rbb.wTb))*0.5
    ra = rotation(rba.wTb)'*(anchor - translation(rba.wTb))
    rb = rotation(rbb.wTb)'*(anchor - translation(rbb.wTb))
    aTb = rotation(rba.wTb)'*rotation(rbb.wTb)
    wRb_a = rotation(rba.wTb)
    wRa_b = rotation(rbb.wTb)*aTb
    setupRigidConstraint(1.0 / dt, rba.wTb, rbb.wTb, ra, rb, wRb_a, wRa_b, c)
end

#== contact manifold ==#

function projectToManifoldVelocity(c::SolverConstraintContactManifold)
    p = c.linearJa*c.bodyA.v - c.linearJb*c.bodyB.v
    p += c.angularJa*c.bodyA.ω + c.angularJb*c.bodyB.ω

    return p
end

function projectToR3Velocity!(c::SolverConstraintContactManifold, dp::Vector3)
    c.bodyA.v += c.linearJa' * dp * c.bodyA.massInv
    c.bodyB.v -= c.linearJb' * dp * c.bodyB.massInv
    c.bodyA.ω += c.bodyA.Iinv * c.angularJa' * dp
    c.bodyB.ω += c.bodyB.Iinv * c.angularJb' * dp
    return c
end

function init(c::SolverConstraintContactManifold)

    Dinv = c.linearJa*c.bodyA.massInv*(c.linearJa')
    Dinv += c.linearJb*c.bodyB.massInv*(c.linearJb')

    Dinv += c.angularJa*c.bodyA.Iinv*c.angularJa'
    Dinv += c.angularJb*c.bodyB.Iinv*c.angularJb'

    c.Dinv = inv(Dinv)

    c.solutionImpulse = zero(c.solutionImpulse)
    c.normalImpulse = zero(c.normalImpulse)
    return c
end

function setupContactManifoldConstraint(dt_inv, points::Vector{ManifoldContact}, separating_axis::Vector3, mu, restitution, c::SolverConstraintContactManifold)
    wTa = c.bodyA.wTb;
    wTb = c.bodyB.wTb;

    npoints = Scalar(length(points))
    localPoints = map(cc->rotate(wTa, cc.ra), points)  # worldspace offset from body A COM
    frictionPatchCentroid = sum(localPoints)/npoints  # worlspace offset from body A COM

    twistMoment = sum(map(p->norm(p-frictionPatchCentroid), localPoints))/npoints

    twistAxis = separating_axis;

    frictionPatchCentroidWorld = translation(wTa) + frictionPatchCentroid
    
    rra = frictionPatchCentroid
    rrb = translation(wTa) + frictionPatchCentroid - translation(wTb)

    frictionx, frictiony, _ = generate_frame(twistAxis)

    c.linearJa = [frictionx frictiony zeros(3)];

    c.linearJb = -c.linearJa;

    c.angularJa = [rra×frictionx rra×frictiony twistAxis*twistMoment]
    c.angularJb = [rrb×frictionx rrb×frictiony -twistAxis*twistMoment];

    c.mu = mu;

    c.contactPoints = map(points) do p
        #onedof = setup1DofConstraint(dt_inv, wTa, wTb, p.ra, p.rb, separating_axis, SolverConstraint1Dof(c.bodyA, c.bodyB))
        onedof = setupContactConstraint(dt_inv, wTa, wTb, p.ra, p.rb, separating_axis, SolverConstraintContact(c.bodyA, c.bodyB))
        manvel = projectToManifoldVelocity(onedof)  + 9.81/dt_inv  #!me

        # add some bounce. project extra velocity to make constraint react stronger 
        #!me restitution is a bit tricky, probably need to rethink how to do it.  if weird stuff happens with contacts this is likely culprit 
        if onedof.normal_constraint.manifoldVelocityToCorrect < 0 && manvel < 0
            onedof.normal_constraint.manifoldVelocityToCorrect = min(onedof.normal_constraint.manifoldVelocityToCorrect, manvel*restitution)
        else
            onedof.normal_constraint.manifoldVelocityToCorrect -= manvel*restitution
        end
        onedof
        #SolverConstraintContactManifoldPoint(0, 0, onedof.Ja, onedof.Jb, onedof.Dinv)
    end

    init(c)
end

#== end Constraint type methods ==#

function solve(c::SolverConstraint)
    # project velocities onto constraint manifold. i.e. this gives us the magnitude of the velocity violating the constraint conditions
    manV = projectToManifoldVelocity(c)

    manV += c.manifoldVelocityToCorrect

    Δimpulse = -c.Dinv * manV  # momentum now

    projectToR3Velocity!(c, Δimpulse)

    c.solutionImpulse += Δimpulse
end

# clamps accumulated solution impulse to be > 0
function solve(cc::SolverConstraintClamped)
    c = unclamped_constraint(cc)
    # project velocities onto constraint manifold. i.e. this gives us the magnitude of the velocity violating the constraint conditions
    manV = projectToManifoldVelocity(c)

    manV += c.manifoldVelocityToCorrect

    Δimpulse = -c.Dinv * manV
    solutionImpulse′ = max.(zero(Scalar), c.solutionImpulse + Δimpulse)
    Δimpulse = solutionImpulse′ - c.solutionImpulse

    projectToR3Velocity!(c, Δimpulse)

    c.solutionImpulse += Δimpulse
end

function solve(c::SolverConstraintContactManifold)
    #c.normalImpulse = zero(c.normalImpulse)
    for cc in c.contactPoints
        si = cc.normal_constraint.solutionImpulse
        solve(cc)
        c.normalImpulse += cc.normal_constraint.solutionImpulse - si
        #SolverConstraintContact(SolverConstraint1Dof(0,0,c.bodyA, c.bodyB, cc.Ja, cc.Jb, c.linearJa[:,3], (cc.Ja⋅cc.Jb)))        
    end

    #==
    // project velocities onto constraint manifold. i.e. this gives us the magnitude of the velocity violating the constraint conditions
    Vector3 manV = projectToManifoldVelocity();

    Vector3 manImpulseDelta = manV;

    manImpulseDelta.TransformVector( m_Dinv );
    manImpulseDelta = -manImpulseDelta;

    Vector3 manImpulse = m_solutionImpulse;

    manImpulse += manImpulseDelta;
    ==#

    # project velocities onto constraint manifold. i.e. this gives us the magnitude of the velocity violating the constraint conditions
    manV = projectToManifoldVelocity(c) + c.manifoldVelocityToCorrect

    Δimpulse = -c.Dinv * manV
    solutionImpulse′ = c.solutionImpulse + Δimpulse

#==

    float xyzw[4];
    manImpulse.ToFloatStar(xyzw);

    float ff = sqrtf( xyzw[0]*xyzw[0] + xyzw[1]*xyzw[1] + xyzw[2]*xyzw[2] );

    float nf4[4];
    float nf;
    Vector3 normalForceMu;
    normalForceMu = m_normalForce * m_mu;
    normalForceMu.ToFloatStar( nf4 );

    nf = nf4[0];
    if( ff > nf )
    {
        float clamper = nf/ff;
        xyzw[0] = xyzw[0]*clamper;
        xyzw[1] = xyzw[1]*clamper;
        xyzw[2] = xyzw[2]*clamper;

        manImpulse.Init( xyzw );

    }
==#

    # clamp
    friction_force = norm(solutionImpulse′)
    println(friction_force, " ff > nf ", c.normalImpulse*c.mu)

    if( friction_force > c.normalImpulse*c.mu )
        clamper = (c.normalImpulse*c.mu)/(friction_force + eps())
        solutionImpulse′ = solutionImpulse′*clamper
    end

#==
    manImpulseDelta = manImpulse - m_solutionImpulse;

    projectToR3Velocity( manImpulseDelta );

    m_solutionImpulse += manImpulseDelta;
    //  m_normalForce += manImpulseDelta;

==#


    Δimpulse = solutionImpulse′ - c.solutionImpulse

    projectToR3Velocity!(c, Δimpulse)

    c.solutionImpulse += Δimpulse

end

##=========== Stepper ==============##

mutable struct RigidBody
    x::Vector3
    m::Scalar
    I::Vector3   # bodyspace
    v::Vector3
    ω::Vector3   # world space
    q::Quaternion{Scalar}
    id::Int64
end

RigidBody(x, m=1.0, inertia=ones(3)) = RigidBody(x, m, inertia,
    zeros(3), zeros(3), Quaternion(1.0, 0, 0, 0), -1)

apply_inertia(q::Quaternion, I, v) = q >>> (I .* inv(q) >>> v)

function ω_implicit(ω₀::Vector3, It, q::Quaternion, dt)
    # one step of newtons method to solve for new angular velocity = f(ω′) = I(ω′-ω)+ω′xIω′*dt = 0
    # df(ω′)/ω′ = I + (1xIω′+ω′xI)*dt
    # df(ω) = I + (ωxI - Iωx1)*dt
    ω = inv(q) >>> ω₀
    Iω = It * ω
    f = ω × Iω * dt
    df = It + (skew(ω) * It - skew(Iω)) * dt

    ω′ = ω - df \ f
    q >>> ω′
end

function integrate_implicit!(rb::RigidBody, dt::Scalar)
    rb.x += rb.v * dt
    rb.ω = ω_implicit(rb.ω, Diagonal(rb.I), rb.q, dt)
    q2 = rb.q + 0.5 * dt * rb.ω * rb.q
    rb.q = q2 / norm(q2)

    return rb
end

function step_world(world, dt::Scalar)
    for rb in world
        integrate_implicit!(rb, dt)
    end
end

function solve_sequential_impulse(bodies, constraints, dt, constraint_iterations)
    solver_bodies = map(to_solver_body, bodies)
    c = map(cᵢ -> to_solver_constraint(cᵢ, solver_bodies, dt), constraints)

    # solve constraints for updated velocity
    for i in 1:constraint_iterations
        for cᵢ in c
            solve(cᵢ)
        end
    end

    map(cᵢ -> cᵢ.solutionImpulse, c)
end

#!me ack, I'm dumb.  J changes so we can't do this as DWtWD isn't static.  
function solve_local_global(bodieso, bodies, constraintso; niter = 4)
    dt = 1.0/30
    constraints = map(cᵢ -> to_solver_constraint(cᵢ, bodies, dt), constraintso)
    J = global_jacobian(constraintso, bodieso, dt, generate_body_mapping(constraintso, bodieso))

    #==
    zeros(ncon*6,nbod*6)
    for (i,c) in enumerate(constraints)
        i6 = i*6-5:i*6
        ja6lin = c.bodyA*6-5:c.bodyA*6-3
        jb6lin = c.bodyB*6-5:c.bodyB*6-3
        ja6ang = ja6lin .+ 3
        jb6ang = jb6lin .+ 3
        J[i6, ja6lin] = linearJa(c) 
        J[i6, ja6ang] = linearJb(c)
        J[i6, jb6lin] = angularJa(c) 
        J[i6, jb6ang] = angularJb(c)
    end
    ==#

    # if we use actual mass we need our x to be converted to momentum somehow.  going to just have mass factor into the local solve right now
    M = 1.0I #inv(global_mass_inv(solver_bodies))

    # ADMM
    x = mapreduce( bod->vcat(bod.v, bod.ω), vcat, bodies )

    ū = zeros(length(constraints)*6)

    D = J
    W = 1.0I 
    ρ = 1.0
    DWtWD = factorize(M + ρ*dt*dt*D'*W'*W*D)
    # working directly on velocity level i.e. x = v
    #    x = x + v * dt + inv(M) * f * dt * dt

    x̃ = copy(x)
    z = D * x

    for i in 1:niter
        Dx = D * x
        Dxu = Dx + ū
        zᵢ = 1
        for (i, c) in enumerate(constraints)

            conrange = i*6-5:i*6
            # z′ = local_update(eᵢ, z[zrangeᵢ, :], Dxu[zrangeᵢ, :], W[zrangeᵢ])
            # local update
            c.bodyA.v = x[c.bodyA.id*6-5:c.bodyA.id*6-3]
            c.bodyA.ω = x[c.bodyA.id*6-2:c.bodyA.id*6] 
            c.bodyB.v = x[c.bodyB.id*6-5:c.bodyB.id*6-3]
            c.bodyB.ω = x[c.bodyB.id*6-2:c.bodyB.id*6]
            c.manifoldVelocityToCorrect = ū[conrange]

            solve(c)

            z[conrange] = projectToManifoldVelocity(c)
            ū[conrange] += Dx[conrange] - z[conrange]
        end
        rhs = M * x̃ + ρ * dt * dt * D' * W' * W * (z - ū)
        x = DWtWD \ sparse(rhs)
    end

    for (i, c) in enumerate(constraints)
        c.bodyA.v = x[c.bodyA.id*6-5:c.bodyA.id*6-3]
        c.bodyA.ω = x[c.bodyA.id*6-2:c.bodyA.id*6] 
        c.bodyB.v = x[c.bodyB.id*6-5:c.bodyB.id*6-3]
        c.bodyB.ω = x[c.bodyB.id*6-2:c.bodyB.id*6]
    end
end

function step_world(bodies, constraints, dt::Scalar; external_force=rb -> zero(Vector3), external_torque=rb -> zero(Vector3), constraint_iterations=8, solver_type=:block_global)
    # Symplectic Euler.  update velocity before constraints and position after constraints
    # integrate velocity
    for rb in bodies
        if rb.m != 0
            rb.v += external_force(rb) * dt
            rb.ω += external_torque(rb) * dt
            rb.ω = ω_implicit(rb.ω, Diagonal(rb.I), rb.q, dt)
        end
    end

    if solver_type == :global
        println("global solver")
        solver_bodies = solve_global(constraints, bodies, dt) |> first

        for (i, bodᵢ) in enumerate(solver_bodies)
            bodies[i].v = bodᵢ.v
            bodies[i].ω = bodᵢ.ω
        end
    elseif solver_type == :block_global
        println("block global solver")
        cg = generate_constraint_graph(bodies, constraints)
        solver_bodies = map(to_solver_body, bodies)
        sc = map(cᵢ -> to_solver_constraint(cᵢ, solver_bodies, dt), constraints)
        precomp = precompute_block_solver(cg, sc)
        solve_global_block(precomp, sc, dt)

    elseif solver_type == :local_global
        println("local global solver")
        #solver_bodies = map(to_solver_body, bodies)
        #c = map(cᵢ -> to_solver_constraint(cᵢ, solver_bodies, dt), constraints)

        solver_bodies = map(to_solver_body, bodies)
        solve_local_global(bodies, solver_bodies, constraints)
    else
        #println("sequential impulse solver")
        solver_bodies = map(to_solver_body, bodies)
        c = map(cᵢ -> to_solver_constraint(cᵢ, solver_bodies, dt), constraints)

        # solve constraints for updated velocity
        for i in 1:constraint_iterations
            for cᵢ in c
                solve(cᵢ)
            end
        end
    end

    for (i, bodᵢ) in enumerate(solver_bodies)
        bodies[i].v = bodᵢ.v
        bodies[i].ω = bodᵢ.ω
    end

    # integrate positions with updated velocities
    for rb in bodies
        rb.x += rb.v * dt
        rb.q = normalize(rb.q + 0.5 * dt * rb.ω * rb.q)
    end
end

#===========  Entities ==============#

abstract type RigidBodyConstraint end

struct PointToPointConstraint <: RigidBodyConstraint
    bodyA::Int64
    bodyB::Int64
    ra::Vector3  # offset from com to pivot in space of body A
    rb::Vector3
end

struct Linear1DofConstraint <: RigidBodyConstraint
    bodyA::Int64
    bodyB::Int64
    ra::Vector3  # offset from com to pivot in space of body A
    rb::Vector3
    axisb::Vector3   # axis of constraint in body B space
end

struct Angular1DofConstraint <: RigidBodyConstraint
    bodyA::Int64
    bodyB::Int64
    perp_axisa::Vector3  # offset from com to pivot in space of body A
    perp_axisb::Vector3
end

Angular1DofConstraint(bodyA, bodyB, axis::Vector3, qa::Quaternion, qb::Quaternion) = Angular1DofConstraint(bodyA, bodyB, ([inv(qa), inv(qb)] .>>> generate_frame(axis)[1:2])...)

struct ContactConstraint <: RigidBodyConstraint
    bodyA::Int64
    bodyB::Int64
    ra::Vector3  # offset from com to pivot in space of body A
    rb::Vector3
    normal::Vector3   # contact normal in body B space
end

# should be between two convex shapes.  all points share same normal ( seperating axis )
struct ContactManifoldConstraint <: RigidBodyConstraint
    bodyA::Int64
    bodyB::Int64
    separating_axis::Vector3
    mu::Scalar
    restitution::Scalar
    contact_points::Vector{ManifoldContact}
end

"""
    rigidly constrain all degrees of freedom
"""
struct RigidConstraint <: RigidBodyConstraint
    bodyA::Int64
    bodyB::Int64
    ra::Vector3  # offset from com to pivot in space of body A
    rb::Vector3
    aRb::Matrix3x3  # relative frame from b to a
end

body_ids(c::T, rb::Vector{RigidBody}) where {T<:RigidBodyConstraint} = (rb[c.bodyA].id, rb[c.bodyB].id)
ndof(c::T) where {T<:RigidBodyConstraint} = 1
ndof(c::T) where {T<:PointToPointConstraint} = 3
ndof(c::T) where {T<:RigidConstraint} = 6

function to_solver_body(rb::RigidBody)
    R = quat_to_dcm(rb.q)'
    minv = 0
    if rb.m == 0
        Iinv_world = zero(Matrix3x3)
    else
        Iinv_world = R * diagm(inv.(rb.I)) * R'
        minv = 1/rb.m
    end

    wTb = create_transform(Array(quat_to_dcm(rb.q)'), rb.x)
    SolverBody(rb.v, rb.ω, wTb, Iinv_world, minv, rb.id)
end

function to_solver_constraint(p2p::PointToPointConstraint, solver_bodies, dt)
    rba = solver_bodies[p2p.bodyA]
    rbb = solver_bodies[p2p.bodyB]
    setupPointToPointConstraint(1.0 / dt, rba.wTb, rbb.wTb, p2p.ra, p2p.rb, SolverConstraint3DofLinear(rba, rbb))
end

function to_solver_constraint(c::Linear1DofConstraint, solver_bodies, dt)
    rba = solver_bodies[c.bodyA]
    rbb = solver_bodies[c.bodyB]
    setup1DofConstraint(1.0 / dt, rba.wTb, rbb.wTb, c.ra, c.rb, rotate(rbb.wTb, c.axisb), SolverConstraint1Dof(rba, rbb))
end

function to_solver_constraint(c::Angular1DofConstraint, solver_bodies, dt)
    rba = solver_bodies[c.bodyA]
    rbb = solver_bodies[c.bodyB]
    setup1DofAngularConstraint(1.0 / dt, rotate(rba.wTb, c.perp_axisa), rotate(rbb.wTb, c.perp_axisb), SolverConstraint1DofAngular(rba, rbb))
end

function to_solver_constraint(c::ContactConstraint, solver_bodies, dt)
    rba = solver_bodies[c.bodyA]
    rbb = solver_bodies[c.bodyB]
    setupContactConstraint(1.0 / dt, rba.wTb, rbb.wTb, c.ra, c.rb, rotate(rbb.wTb, c.normal), SolverConstraintContact(rba, rbb))
end

function to_solver_constraint(c::ContactManifoldConstraint, solver_bodies, dt)
    rba = solver_bodies[c.bodyA]
    rbb = solver_bodies[c.bodyB]
    setupContactManifoldConstraint(1.0/dt, c.contact_points, c.separating_axis, c.mu, c.restitution, SolverConstraintContactManifold(rba, rbb) )
end

function to_solver_constraint(c::RigidConstraint, solver_bodies, dt)
    rba = solver_bodies[c.bodyA]
    rbb = solver_bodies[c.bodyB]

    # the b in wTb is for "body" not body "b". everywhere else here it's for body "b"
    #wRb_a = rotation(rba.wTb) * c.aRb
    #wRa_b = rotation(rbb.wTb) * c.aRb'
    aTb = rotation(rba.wTb)'*rotation(rbb.wTb)
    wRb_a = rotation(rba.wTb)*c.aRb
    wRa_b = rotation(rbb.wTb)*c.aRb'*aTb
    setupRigidConstraint(1.0 / dt, rba.wTb, rbb.wTb, c.ra, c.rb, wRb_a, wRa_b, SolverConstraintRigid(rba, rbb))
end


function forcepsd(A, diagfactor=1.0e-8)
    for i in 1:size(A)[1]
        A[i, i+1:end] = A[i+1:end, i]
    end
    A = A + Diagonal(A) * diagfactor
    A
end
forcepsd(A::BlockArray{N}, diagfactor=1.0e-8) where N = BlockArray{N}(forcepsd(A.A, diagfactor))
forcepsd(A::T, diagfactor=1.0e-8) where T<:StaticArray = T(forcepsd(Array(A), diagfactor))

#!me this is the one that the C++ code use.  forcepsd is better.  actually we want a little of both if we use 1/mass = 0.0 for infinitely massy objects ( the ground ).  then we want to add 1.0e-16 at least
function force_sym(A, diagfactor=1.0e-9)
    for i in 1:size(A)[1]
        A[i, i+1:end] = A[i+1:end, i]
    end
    A = A + diagm(ones(size(A)[1]) * diagfactor)
    A
end
force_sym(A::BlockArray{N}, diagfactor=1.0-9) where N = BlockArray{N}(force_sym(A.A, diagfactor))
force_sym(a::T, diagfactor=1.0-9) where T<:Number = a + diagfactor
force_sym(A::T, diagfactor=1.0e-9) where T<:StaticArray = T(force_sym(Array(A), diagfactor))

#==
end # module

module BlockPSDSolver

using RBD, LinearAlgebra, SparseArrays, StaticArrays, ReferenceFrameRotations 
==#

function generate_body_mapping(constraints::Vector{T}, bodies::Vector{RigidBody}) where {T}
    rbids = (sort ∘ collect)(mapreduce(x -> body_ids(x, bodies), ∪, constraints; init=Set{Int64}()))

    id2active = zeros(Int64, last(rbids))
    id2active[rbids] = 1:length(rbids)

    active2original = Vector{Int64}(undef, length(rbids))
    for i in 1:length(rbids)
        j = 1
        while bodies[i].id != rbids[j]
            j += 1
        end
        active2original[j] = i
    end

    (id2active, active2original)
end

function global_jacobian(constraints, bodies::Vector{RigidBody}, dt, body_map) # where {T<:RigidBodyConstraint}
    id2active, i2original =  body_map
    solver_bodies = map(to_solver_body, bodies)
    c = map(cᵢ -> to_solver_constraint(cᵢ, solver_bodies, dt), constraints)

    I = zeros(Int64, sum(map(ndof, constraints)) * 12)
    row = 1
    for cᵢ in constraints
        for dofi in 1:ndof(cᵢ)
            I[((row-1)*12+1):(row*12)] .= row
            row += 1
        end
    end

    J = map(constraints) do cᵢ
        a, b = id2active[[body_ids(cᵢ, bodies)...]]
        a = (a - 1) * 6
        b = (b - 1) * 6
        repeat([(+).(a, 1:6)..., (+).(b, 1:6)...], ndof(cᵢ))
    end

    # either 1 or two vectors
    V = map(c) do cᵢ
        nd = ndof(cᵢ)
        Jax = linearJa(cᵢ)
        Jbx = linearJb(cᵢ)
        Jaω = angularJa(cᵢ)
        Jbω = angularJb(cᵢ)
        if nd == 1
            [Jax, Jaω, Jbx, Jbω]
        else
            map(1:nd) do i
                [Jax[i, :], Jaω[i, :], Jbx[i, :], Jbω[i, :]]
            end
        end

    end

    ∂C∂x = sparse(I, reduce(vcat, J), reduce(vcat, reduce(vcat, reduce(vcat, V))))
end

global_jacobian(constraints::Vector{T}, bodies::Vector{RigidBody}, dt=1.0/60) where {T<:RigidBodyConstraint} = 
    global_jacobian(constraints, bodies, dt, generate_body_mapping(constraints, bodies))

function global_mass_inv(solver_bodies, i2original)
    M⁻¹ = sparse(diagm(zeros(6 * length(i2original))))

    for (i, b) in enumerate(solver_bodies[i2original])
        mi = (i - 1) * 6 + 1
        M⁻¹[mi:(mi+2), mi:(mi+2)] = diagm(repeat([b.massInv], 3))
        Ii = mi + 3
        M⁻¹[Ii:(Ii+2), Ii:(Ii+2)] = b.Iinv
    end

    M⁻¹
end

global_mass_inv(solver_bodies) = global_mass_inv(solver_bodies, collect(1:length(solver_bodies)))

"""
    inefficient full matrix from scratch solve.  for reference/debugging purposes.
"""
function global_system_matrix(constraints::Vector{T}, bodies::Vector{RigidBody}, dt) where {T<:RigidBodyConstraint}
    body_map = generate_body_mapping(constraints, bodies)
    _, i2original = body_map

    ∂C∂x = global_jacobian(constraints, bodies, dt, body_map)

    solver_bodies = map(to_solver_body, bodies)

    M⁻¹ = global_mass_inv(solver_bodies, i2original)    
    
    #display(Matrix(∂C∂x*M⁻¹*(∂C∂x') |> forcepsd))

    ∂C∂x * M⁻¹ * (∂C∂x')
end

function solve_global(constraints::Vector{T}, bodies::Vector{RigidBody}, dt) where {T<:RigidBodyConstraint}

    A = global_system_matrix(constraints, bodies, dt) |> forcepsd |> factorize

    solver_bodies = map(to_solver_body, bodies)
    c = map(cᵢ -> to_solver_constraint(cᵢ, solver_bodies, dt), constraints)

    b = -mapreduce(rhs, vcat, c; init=Vector{Scalar}([]))

    # solve for constraint impulses 
    λ = A \ b

    i = 1
    for cᵢ in c
        nd = ndof(cᵢ)
        if nd == 1
            projectToR3Velocity!(cᵢ, λ[i])
            cᵢ.solutionImpulse += λ[i]
        else
            #println(λ[i:(i+nd-1)])
            projectToR3Velocity!(cᵢ, λ[i:(i+nd-1)])
            cᵢ.solutionImpulse += λ[i:(i+nd-1)]
        end
        i += nd
    end

    (solver_bodies, map(cᵢ -> cᵢ.solutionImpulse, c))
end

function generate_constraint_graph(bodies::Vector{RigidBody}, c::Vector{T}) where {T<:RigidBodyConstraint}
    body2constraint = ((bᵢ, Set{Int64}()) for bᵢ in mapreduce(cᵢ -> collect(body_ids(cᵢ, bodies)), vcat, c)) |> Dict

    for (i, cᵢ) in enumerate(c)
        for bᵢ in body_ids(cᵢ, bodies)
            body2constraint[bᵢ] = body2constraint[bᵢ] ∪ i
        end
    end

    constraint_graph = map(i -> Vector{Int64}(), 1:length(c))
    # cross connect each constraint attached to same body
    for (bodyᵢ, connectedset) in body2constraint
        connectedc = collect(connectedset)
        for i in 1:length(connectedc)
            a = connectedc[i]
            for j in (i+1):length(connectedc)
                b = connectedc[j]
                constraint_graph[a] = vcat(constraint_graph[a], b)
                constraint_graph[b] = vcat(constraint_graph[b], a)
            end
        end
    end
    constraint_graph
end

function insert_sorted!(tupv::Vector{Tuple{Int64,Int64}}, el::Tuple{Int64,Int64})
    insert!(tupv, searchsortedfirst(tupv, lt=((a, b), (c, d)) -> (b << 32 | a) < (d << 32 | c), el), el)
end

function SmallerCol(r, c)
    if (c < r)
        return (r, c)
    else
        return (c, r)
    end
end

swap(ij) = (ij[2], ij[1])

function reduce_LDLt(constraint_graph, constraints; diagfactor=1e-3)

    LDLt = Dict{Tuple{Int64,Int64},Array{Scalar,2}}()
    # populate hash table with entries we have
    # each row reduction is like collapsing an edge in the constraint graph.  It will alter existing cross-terms as well as create new ones ( fillin )
    for i in 1:length(constraint_graph)
        rigidᵢ = constraints[i]

        #LDLt[(i,i)] = diagonal_term(rigidᵢ) + Diagonal(vcat(repeat([1.0/rigidᵢ.bodyA.massInv+1.0/rigidᵢ.bodyB.massInv], 6), ))*1e-6
        LDLt[(i, i)] = diagonal_term(rigidᵢ) + Diagonal(ones(6) * diagfactor)

        for j in constraint_graph[i]
            if i < j
                LDLt[(j, i)] = cross_term(constraints[j], constraints[i])
            end
        end
    end

    # go through reduction for each row in turn.
    # we will use a sorted queue for fillin.  Add fillin here and pull from the back until we've got all the columns for this current col/row reduction
    fillinq = Vector{Tuple{Int64,Int64}}()
    allconnect = Vector{Tuple{Int64,Int64}}()

    reduced_entry = Vector{Array{Scalar,2}}()

    # each row/col reduction is like collapsing an edge in the constraint graph.  It will alter existing cross-terms as well as create new ones ( fillin )
    # i is column, k is row
    for ireduce in 1:length(constraint_graph)
        # can't use Dinv from the constraint because it may be changed during reduction.
        # might be good to track when it's actually changed to avoid expensive inverse. maybe as part of the fillin
        Dii_inv = inv(LDLt[(ireduce, ireduce)])
        #rigidᵢ = constraints[ireduce]
        allconnect = empty!(allconnect)

        while (length(fillinq) > 0 && fillinq[1][2] == ireduce)  #!me did I sort by row or column as most significant "digit"?
            insert_sorted!(allconnect, popfirst!(fillinq))
        end

        for iconnect in 1:length(constraint_graph[ireduce])
            jconnect = constraint_graph[ireduce][iconnect]
            # any col < the one we are reducing has already be reduced
            if jconnect > ireduce
                insert_sorted!(allconnect, SmallerCol(jconnect, ireduce))
            end
        end

        for i in 1:length(allconnect)
            r = allconnect[i][1]
            ir = SmallerCol(r, ireduce)
            Eir = LDLt[ir]

            DEt = Dii_inv * Eir'

            # store reduction col entry in temp buffer
            push!(reduced_entry, DEt')

            for j in i:length(allconnect)
                c = allconnect[j][1]
                Eic = LDLt[SmallerCol(c, ireduce)]

                EDEt = Eic * DEt

                # store reduction.  possibly creating new fillin
                rc = SmallerCol(r, c)
                if !haskey(LDLt, rc)
                    insert_sorted!(fillinq, rc)
                    LDLt[rc] = -EDEt
                else
                    Ecr = LDLt[rc]
                    LDLt[rc] = Ecr - EDEt
                end
            end
        end

        # copy over reduction row/col entry from temp buffer
        for i in 1:length(allconnect)
            r = allconnect[i][1]
            ir = SmallerCol(r, ireduce)

            LDLt[ir] = popfirst!(reduced_entry)
        end
    end

    LDLt
end

function assemble_block_stream(LDLt::T) where {T<:Dict}
    # want them stored row first swap row and column for sorting.  Need to swap back after
    allconnect = mapreduce(swap, insert_sorted!, keys(LDLt); init=Vector{Tuple{Int64,Int64}}())

    block_stream = Vector{Array{Scalar,2}}()
    block_ij = Vector{Tuple{Int64,Int64}}()

    while !isempty(allconnect)
        ij = swap(popfirst!(allconnect))
        push!(block_stream, LDLt[ij])
        push!(block_ij, ij)
    end

    block_stream_upper = Vector{Array{Scalar,2}}()
    block_ij_upper = Vector{Tuple{Int64,Int64}}()
    # upper.  we want to iterate in reverse order on the Lt matrix. it's transposed, so sorting col major ==> row major for the transpose
    allconnect = reduce(insert_sorted!, keys(LDLt); init=empty!(allconnect))

    while !isempty(allconnect)
        ij = popfirst!(allconnect)
        pushfirst!(block_stream_upper, LDLt[ij]')
        pushfirst!(block_ij_upper, swap(ij))
    end

    (block_stream, block_ij, block_stream_upper, block_ij_upper)
end

function precompute_block_solver(constraint_graph, solver_constraints)
    LDLt = reduce_LDLt(constraint_graph, solver_constraints)
    assemble_block_stream(LDLt)
end

function solve_global_block((blocks, block_ij, blocks_upper, block_ij_upper), rigid_constraints, dt)
    diagfudge = diagm(ones(6) * 1e-8) * 0

    b = -map(rhs, rigid_constraints)

    # solving LDLt x = b
    # first notice: L [DLt x = z] = b
    # solve Lz = b  for z
    # then DLt x = z for x
    # note: Lt x = inv(D)*z

    D = Vector{Array{Scalar,2}}()

    current_row = 1
    iblock_stream = 1
    while iblock_stream < length(block_ij) + 1
        bsum = b[current_row]
        # diagonal is at the end, so iterate over elements in this row until same index, then we have diagonal.
        while block_ij[iblock_stream][2] != current_row
            col = block_ij[iblock_stream][2]
            @assert(col < current_row)
            @assert(block_ij[iblock_stream][1] == current_row)
            bsum = bsum - blocks[iblock_stream] * b[col]
            iblock_stream = iblock_stream + 1
        end

        # block stream should point at diagonal.  this inv is suboptimal
        #b[current_row] = inv(blocks[iblock_stream] + diagfudge) * bsum
        b[current_row] = bsum
        push!(D, inv(blocks[iblock_stream] + diagfudge))

        iblock_stream = iblock_stream + 1
        current_row = current_row + 1
    end

    # apply D⁻¹ to y
    for (i, Dᵢ) in enumerate(D)
        b[i] = Dᵢ * b[i]
    end

    # now solve Lt x = y   ( we already scaled y by inv(D) )
    # reuse b data for solution
    current_row = length(rigid_constraints)
    iblock_stream = 1
    while iblock_stream < length(block_ij_upper) + 1
        bsum = b[current_row]
        # diagonal is at the end, so iterate over elements in this row until same index, then we have diagonal.
        while block_ij_upper[iblock_stream][2] != current_row
            col = block_ij_upper[iblock_stream][2]
            @assert(col > current_row)
            @assert(block_ij_upper[iblock_stream][1] == current_row)
            bsum = bsum - blocks_upper[iblock_stream] * b[col]
            iblock_stream = iblock_stream + 1
        end

        b[current_row] = bsum
        iblock_stream = iblock_stream + 1
        current_row = current_row - 1
    end

    for i in 1:length(rigid_constraints)
        c = rigid_constraints[i]
        dimpulse = b[i]
        #println(dimpulse)
        projectToR3Velocity!(c, dimpulse)
        c.solutionImpulse = c.solutionImpulse + dimpulse
    end

    return 1
end

function LDLt_to_Arrays((blocks, block_ij, blocks_upper, block_ij_upper))
    n = reduce(max, mapreduce(collect, vcat, block_ij)) * 6
    L = zeros(n, n)
    D = zeros(n, n)
    Lt = zeros(n, n)
    for i in 1:length(blocks)
        r = block_ij[i][1]
        c = block_ij[i][2]
        if r == c
            D[(r-1)*6+1:r*6, (c-1)*6+1:c*6] = blocks[i]
            L[(r-1)*6+1:r*6, (c-1)*6+1:c*6] = diagm(ones(6))
            Lt[(r-1)*6+1:r*6, (c-1)*6+1:c*6] = diagm(ones(6))
        else
            L[(r-1)*6+1:r*6, (c-1)*6+1:c*6] = blocks[i]
        end
    end
    for i in 1:length(blocks_upper)
        r = block_ij_upper[i][1]
        c = block_ij_upper[i][2]
        if r != c
            Lt[(r-1)*6+1:r*6, (c-1)*6+1:c*6] = blocks_upper[i]
        end
    end
    (L, D, Lt)
end

function column_count(colj)
    (n, _, colsum) = reduce(((c, j, acc), el) -> el == j ? (c + 1, j, acc) : (c + 1, j + 1, vcat(acc, c)), colj; init=(1, 1, [1]))
    vcat(colsum, n)
end

function cholsol(Lblocks::Vector{Tuple{Tuple{T,T},Matrix{F}}}, x::Vector{V}, removed=[]) where {T<:Integer,F<:Number,V<:AbstractVector{F}}
    succ = x -> x + 1
    second = x -> x[2]
    n = reduce(max, first.(first.(Lblocks)))+1
    x = copy(x)

    Lp = column_count(succ.(second.(first.(Lblocks))))
    Lx = second.(Lblocks) |> collect
    Li = succ.(first.(first.(Lblocks))) |> collect
    # column-wise multiplication
    for j in 1:n
        #x[j] /= Lx[Lp[j]];
        x[j] = inv(Lx[Lp[j]]) * x[j]
        p = Lp[j] + 1
        while p < Lp[j+1]
            #x[Li[p]] -= Lx[p] * x[j];
            x[Li[p]] = x[Li[p]] - Lx[p] * x[j]
            p = succ(p)
        end
    end
    #cs_ltsolve (N->L, x) ;		/* x = L'\x */
    # row-wise multiplication
    for j in n:-1:1
        if isempty(removed) || removed[j] == 0
            p = Lp[j] + 1
            while p < Lp[j+1]
                # x[j] -= Lx[p] * x[Li[p]];
                x[j] = x[j] - Lx[p]' * x[Li[p]]
                p = succ(p)
            end
            #!me clearly we should store D pre-inverted 
            x[j] = inv(Lx[Lp[j]])' * x[j]
        else
            x[j] = zero(x[j])
        end
    end

    x
end


function LBlock_to_L( Lblocks::Vector{Tuple{Tuple{T,T},Matrix{F}}} ) where {T<:Integer,F<:Number}
    n = reduce(max, first.(first.(Lblocks)))+1

    L = zeros(n*6, n*6)
    for ((i,j), Lx) in Lblocks
        L[(i*6+1):(i+1)*6, (j*6+1):(j+1)*6] = Lx
    end
    L
end

function cholupdate(L, x, first_nonzero = 1)
    L = deepcopy(L)
    n = length(x)
    for k = first_nonzero:n
        r = sqrt(L[k, k] * L[k, k]' + x[k] * x[k]')
        c = r' / L[k, k]'
        s = x[k]' / L[k, k]'
        L[k, k] = r
        if k < n
            L[(k+1):n, k] = (L[(k+1):n, k] + x[(k+1):n]*s) / c
            x[(k+1):n] = x[(k+1):n]*c - L[(k+1):n, k]*s
        end
    end
    L
end

function choldowndate(L, x)
    L = deepcopy(L)
    n = length(x)
    for k = 1:n
        r = sqrt(L[k, k] * L[k, k]' - x[k] * x[k]')
        c = r' / L[k, k]'
        s = x[k]' / L[k, k]'
        L[k, k] = r
        if k < n
            L[(k+1):n, k] = (L[(k+1):n, k] - x[(k+1):n]*s) / c
            x[(k+1):n] = c * x[(k+1):n] - L[(k+1):n, k]*s
        end
    end
    L
end

# format records like this "^.*LogTemp: ((i,j), [6x6 array])"
function parse_log(log::S) where S<:String
    log′ = filter( x->length(x) > 1, split(replace(log, r".*LogTemp:"=>""), "\n\n"))
    ij = map( x->(eval(Meta.parse(x[3:end-1]))), (x->x[2]).(filter( (isodd∘first), enumerate(log′)|>collect)))
    el = map( x->eval(Meta.parse(x)), (x->x[2]).(filter( (iseven∘first), enumerate(log′)|>collect)))
    zip(ij, el)
end

function parse_body_log(log::S) where S<:String
    log′ = prod(filter( x->length(x) > 1, split(replace(log, r".*LogTemp:"=>""), "\n\n")))
    Meta.parse("["*prod(log′)*"]") |> eval
end


function remove_constraint(L::BlockArray{6}, i)
    x = L[:,i]
    x[i] = zeros(6,6)
    L = cholupdate(L, x, i+1)
    L[1:end .!= i, 1:end .!= i]
end

function remove_constraint_x6(L::BlockArray{6}, i)
    x = L[:,i]
    x[i] = zeros(6,6)
    for j in 1:6
        L = BlockArray{6}(cholupdate(L.A, x.A[:,j], i*6))
    end
    L[1:end .!= i, 1:end .!= i]
end

function remove_LLt(L::BlockArray{6}, i)
    Lr = deepcopy(L)
    if i != size(L)[1]
        x = L[i+1:end, i]
        # L̄33*L̄33'=L33*L33'+x*x'
        L33 = L[i+1:end, i+1:end].A * L[i+1:end, i+1:end].A' + x.A * x.A' |> choln |> BlockArray{6}
        Lr[i+1:end, i+1:end] = L33
    end
    Lr[1:end.!=i, 1:end.!=i]
end

end



using .RBD
using LinearAlgebra, StaticArrays, ReferenceFrameRotations
drawit = true
using MeshCat, GeometryBasics, CoordinateTransformations

dt = 1.0/60.0

function add_box(offset, box_radius, id; rot=Quaternion(1.0, 0, 0, 0), is_static=false)
    x = offset
    rb = RigidBody(x)
    rb.q = rot
    rb.ω = [0.0, 0, 0]
    rb.x = x
    rb.v = [0.0, 0.0, 0.0]
    rb.m = is_static ? 0 : 10
    rb.I = inertia_tensor(Vector3(2 * box_radius), rb.m)
    rb.id = id
    symname = Symbol("box", id)
    global drawit
    if drawit
        setobject!(vis[symname], Rect(-Vec(box_radius), Vec(box_radius) * 2))
        #setobject!(vis[symname], Sphere(Point(rb.x...), box_radius[1]))
        #atframe(anim, 0) do
        #    settransform!(vis[symname], compose(Translation(rb.x...), LinearMap(quat_to_dcm(rb.q)')))
        #end
    end
    (rb, symname)
end

function add_sphere(offset, radius, id; rot=Quaternion(1.0, 0, 0, 0), is_static=false)
    x = offset
    rb = RigidBody(x)
    rb.q = rot
    rb.ω = [0.0, 0, 0]
    rb.x = x
    rb.v = [0.0, 0.0, 0.0]
    rb.m = is_static ? 0 : 10
    rb.I = inertia_tensor(Vector3((2 * radius) .* ones(3)), rb.m)
    rb.id = id
    symname = Symbol("sphere", id)
    global drawit
    if drawit
        setobject!(vis[symname], Sphere(Point(0.0,0,0), radius))
        #atframe(anim, 0) do
        #    settransform!(vis[symname], compose(Translation(rb.x...), LinearMap(quat_to_dcm(rb.q)')))
        #end
    end
    (rb, symname)
end

function constrain_shapes(rb, rb2, id, id2, p)
    #p2p = PointToPointConstraint(id, id2, p-rb.x, p-rb2.x)
    #c1 = Angular1DofConstraint(id, id2, Vector3(0,0,1), rb.q, rb2.q)
    #c2 = Angular1DofConstraint(id, id2, Vector3(1,0,0), rb.q, rb2.q)
    #c3 = Angular1DofConstraint(id, id2, Vector3(0,1,0), rb.q, rb2.q)
    #[p2p, c1, c2, c3]
    to_body(p, rb) = inv_rotation(rb.q)*p*rb.q |> imag
    #to_body(p, rb) = p

    aFb = Matrix3x3(Array(quat_to_dcm(rb.q)*inv(quat_to_dcm(rb2.q))))

    cr = RigidConstraint(id, id2, to_body(p - rb.x, rb), to_body(p - rb2.x, rb2), aFb)
    [cr]
end

if drawit
    vis = Visualizer()
    anim = Animation()
end

ground_id = 1
ground_radius = Vector3(10,10,0.01)
ground, groundname = add_box(Vector3(0,0,-2-ground_radius[3]), ground_radius, ground_id; is_static = true)

use_sphere = false
sphere_radius = 0.5
box_radius = Vector3(0.05,0.2,0.4)
offset = Vector3(0,0.0,0)

nbox = 2
bodies = Vector{RigidBody}([ground])
drawbod = [(groundname, ground_radius, ground)]
constraints = Vector{RigidBodyConstraint}([])
for i in 2:nbox
    if !use_sphere
        rb, symname = add_box(Vector3(0,2*(i-1)*box_radius[2]-nbox*box_radius[2],0), box_radius, i)
    else
        rb, symname = add_sphere(Vector3(0,i-2,0), sphere_radius, i)
    end
    if rb.m == 0
        rb.ω = Vector3(0.0,0,0)
    else
        if !use_sphere
            rb.ω = Vector3(0.0,0,10)
            rb.v = Vector3(0.0,0,0)
        else
            rb.ω = Vector3(0.0,0,0)
            rb.v = Vector3(0.0,0,-2)
            rb.x = Vector3(0,0,4/60+ground.x[3]+ground_radius[3]+sphere_radius)
        end
    end
    global bodies
    global drawbod
    global constraints
    bodies = vcat(bodies, [rb])
    drawbod = vcat(drawbod, (symname, nothing, rb))

    if i > 2
        constraints = vcat(constraints, constrain_boxes(bodies[i-1], rb, i-1, i, rb.x+Vector3(0,box_radius[2],0)))
    end

    if i==2
        if !use_sphere
            cmc = ContactManifoldConstraint( 
                i, ground_id, Vector3(0,0,1), 0.01, 0.0,
                map(
                    signit -> ManifoldContact(box_radius .* signit, Vector3(-box_radius[1], -box_radius[2], ground_radius[3])),
                    [[x,y,z] for x in [1,-1] for y in [1,-1] for z in [1,-1]]))
        else
            cmc = ContactManifoldConstraint( 
                i, ground_id, Vector3(0,0,1), 0.7, 1.0,
                    [ManifoldContact(Vector3(rb.x[1], rb.x[2], -sphere_radius), Vector3(rb.x[1], rb.x[2], ground_radius[3]))])
        end
        constraints = vcat(cmc, constraints)
    end
end



#==
    nbox = 1
    box_radius = Vector3(1,1,1)
    rb, symname = add_box(Vector3(0,0,8), box_radius, 1)
    bodies = [rb]
    drawbod = [(symname, box_radius, rb)]
    constraints = [ContactConstraint(1, ground)]
    ==#
if drawit
    for i in 1:length(drawbod)
        rb = drawbod[i][3]
        atframe(anim, 0) do
            settransform!(vis[drawbod[i][1]], compose(Translation(rb.x...), LinearMap(quat_to_dcm(rb.q)')))
        end
    end
end

for i in 1:360
    #step_world(bodies, constraints, dt; use_global_solver=(length(constraints)>1), constraint_iterations=32 )
    step_world(bodies, constraints, dt; solver_type=:local, constraint_iterations=2, external_force = (_)->[0,0,-9.81] )
    if drawit
        atframe(anim, i) do
            for i in 1:length(drawbod)
                rb = drawbod[i][3]
                settransform!(vis[drawbod[i][1]], compose(Translation(rb.x...), LinearMap(quat_to_dcm(rb.q)')))
            end
        end
    end
end

# DRAW
if drawit
    setanimation!(vis, anim)
end

sleep(3)  # seems to need a bit of time to connect to renderer

#==
drawbod = []

dt = 1.0
box_radius = Vector3(0.5, 0.5, 0.5)
rb0, symname0 = add_box(Vector3(box_radius[1] * 0, box_radius[2] * 0, 0), box_radius, 1; rot=angleaxis_to_quat(0.2*π, [0,0,1]))
drawbod = vcat(drawbod, (symname0, box_radius, rb0))
rb1, symname1 = add_box(Vector3(box_radius[1]*2.0, box_radius[2]*0, 0), box_radius, 2)
drawbod = vcat(drawbod, (symname1, box_radius, rb1))

anchor = 0.5 * (rb0.x + rb1.x)
c0 = constrain_boxes(rb0, rb1, 1, 2, anchor)

#rb1.v = Vector3(0.0, 0.5, 0)
rb0.q = angleaxis_to_quat(0.0*π/2, [0,0,1])

bodies = [rb0, rb1]
constraints = c0
==#


#==
dt = 0.0167
box_radius = Vector3(0.2, 0.4, 0.5)
rb0, symname0 = add_box(Vector3(box_radius[1] * 0, box_radius[2] * -2, 0), box_radius, 1)
drawbod = vcat(drawbod, (symname0, box_radius, rb0))
#rb1, symname1 = add_box(Vector3(box_radius[1]*0, box_radius[2]*0, 0), box_radius, 1; rot=angleaxis_to_quat(π/6, [1,0,0]))
rb1, symname1 = add_box(Vector3(box_radius[1] * 0, box_radius[2] * 0, 0), box_radius, 2)
drawbod = vcat(drawbod, (symname1, box_radius, rb1))
rb2, symname2 = add_box(Vector3(box_radius[1] * 2, box_radius[2] * 0, 0), box_radius, 3)
drawbod = vcat(drawbod, (symname2, box_radius, rb2))
rb3, symname3 = add_box(Vector3(box_radius[1] * 2, box_radius[2] * -2, 0), box_radius, 4)
drawbod = vcat(drawbod, (symname3, box_radius, rb3))
rb1.v = Vector3(10.0, 0, 0)
anchor = 0.5 * (rb0.x + rb1.x)
c0 = constrain_boxes(rb0, rb1, 1, 2, anchor)
anchor = 0.5 * (rb1.x + rb2.x)
c1 = constrain_boxes(rb1, rb2, 2, 3, anchor)
anchor = 0.5 * (rb2.x + rb3.x)
c2 = constrain_boxes(rb2, rb3, 3, 4, anchor)
anchor = 0.5 * (rb3.x + rb0.x)
c3 = constrain_boxes(rb3, rb0, 4, 1, anchor)

#rb1.x = rb1.x - Vector3(0.1,0,0)

bodies = [rb0, rb1, rb2, rb3]
constraints = vcat(c0, c1, c2, c3)
==#

#==
sbodies = to_solver_body.(bodies)
sc = (x -> to_solver_constraint(x, sbodies, dt)).(constraints)

cg = generate_constraint_graph(bodies, constraints)
precomp = precompute_block_solver(cg, sc);
solve_global_block(precomp, sc, dt)
==#

#==

for i in 1:300
    println("======== Start Frame ", i,  " ===========")
    step_world(bodies, constraints, dt; solver_type=:block_global, constraint_iterations=32 )
    if drawit
        atframe(anim, i) do
            for i in 1:length(drawbod)
                rb = drawbod[i][3]
                settransform!(vis[drawbod[i][1]], compose(Translation(rb.x...), LinearMap(quat_to_dcm(rb.q)')))
            end
        end
    end
end


# DRAW
if drawit
    setanimation!(vis, anim)
end
sleep(3)  # seems to need a bit of time to connect to renderer


trip_to_matrix(Atrip) = sparse([i + 1 for (i, j, v) in Atrip], [j + 1 for (i, j, v) in Atrip], [v for (i, j, v) in Atrip]) |> collect


function solve_lotri_block(L, b)
    nblock = div(size(L)[1], 6)
    x = [zeros(6, 6) for i in 1:nblock]
    for i in 1:nblock
        x[i] = b[(i-1)*6+1:i*6, 1:6]
        for j in 1:i-1
            x[i] -= L[(i-1)*6+1:i*6, (j-1)*6+1:j*6] * x[j]
        end
        D = L[(i-1)*6+1:i*6, (i-1)*6+1:i*6]
        x[i] = inv(D) * x[i]
    end
    x
end
==#

#==
Lx = [((0, 0), [1.456365 0.0 0.0 0.0 0.0 0.0; 0.659175 0.496476 0.0 0.0 0.0 0.0; 0.0 0.0 1.211866 0.0 0.0 0.0; 0.0 0.0 0.0 1.21013 0.0 0.0; 0.0 0.0 0.0 0.0 1.438737 0.0; 0.0 0.0 0.0 0.0 0.0 1.732339]), ((1, 0), [0.179453 0.483407 0.0 -0.0 -0.0 -0.346352; -0.091166 0.20142 0.0 -0.0 -0.0 -0.0; 0.0 0.0 0.275729 0.241861 0.0 0.0; 0.0 0.0 0.483028 0.604652 0.0 0.0; 0.0 0.0 0.341451 0.0 0.719021 0.0; -0.276973 -1.208519 0.0 0.0 0.0 0.865881]), ((5, 0), [-0.179453 -0.483407 0.0 0.0 0.0 -0.346352; 0.091166 -0.20142 0.0 0.0 0.0 0.0; 0.0 0.0 -0.275729 0.241861 0.0 0.0; 0.0 0.0 0.483028 -0.604652 0.0 0.0; 0.0 0.0 0.341451 0.0 -0.719021 0.0; -0.276973 -1.208519 0.0 0.0 0.0 -0.865881]), ((1, 1), [1.446191 0.0 0.0 0.0 0.0 0.0; -0.056015 1.394739 0.0 0.0 0.0 0.0; 0.0 0.0 1.448081 0.0 0.0 0.0; 0.0 0.0 -0.192963 1.620054 0.0 0.0; 0.0 0.0 -0.065016 -0.10955 1.793452 0.0; 0.645704 0.182355 0.0 0.0 0.0 1.435321]), ((2, 1), [-0.069147 0.0 0.0 0.0 0.0 0.031107; 0.0802 -0.071698 0.0 0.0 0.0 0.182042; 0.0 0.0 -0.069057 -0.016195 -0.117866 0.0; 0.0 0.0 0.202118 -0.427087 0.007327 0.0; 0.0 0.0 0.0 -0.039005 -0.576811 0.0; -0.414883 0.0 0.0 0.0 0.0 -0.85842]), ((5, 1), [0.103154 0.058081 0.0 0.0 0.0 -0.286493; 0.057372 0.035047 0.0 0.0 0.0 -0.182263; 0.0 0.0 0.012105 -0.003038 0.052934 0.0; 0.0 0.0 0.009017 0.076534 -0.091636 0.0; 0.0 0.0 -0.065016 -0.094612 0.220901 0.0; 0.237018 0.156423 0.0 0.0 0.0 -0.675145]), ((2, 2), [0.441872 0.0 0.0 0.0 0.0 0.0; -0.000265 0.525631 0.0 0.0 0.0 0.0; 0.0 0.0 0.514621 0.0 0.0 0.0; 0.0 0.0 0.01536 1.113944 0.0 0.0; 0.0 0.0 -0.133337 -0.009322 1.310675 0.0; -0.004492 0.360597 0.0 0.0 0.0 1.400691]), ((4, 2), [0.226447 0.228297 0.0 0.0 0.0 0.370313; 0.000114 0.190248 0.0 0.0 0.0 -0.048977; 0.0 0.0 0.194318 -0.265259 0.019768 0.0; 0.0 0.0 0.0 0.656862 0.0 0.0; 0.0 0.0 -0.402036 0.011806 0.748375 0.0; 0.000343 0.570743 0.0 0.0 0.0 0.923967]), ((5, 2), [0.036366 0.091405 0.0 0.0 0.0 -0.168439; 0.021844 0.05915 0.0 0.0 0.0 -0.109865; 0.0 0.0 0.013653 -0.003692 0.024594 0.0; 0.0 0.0 -0.017369 0.028216 -0.039817 0.0; 0.0 0.0 0.038892 -0.025644 0.098357 0.0; 0.084751 0.218997 0.0 0.0 0.0 -0.399669]), ((6, 2), [0.226447 0.228297 0.0 0.0 0.0 0.370313; 4.6e-5 0.076099 0.0 0.0 0.0 -0.233771; 0.0 0.0 0.11391 -0.262898 0.169443 0.0; 0.0 0.0 0.0 0.656862 0.0 0.0; 0.0 0.0 -0.402036 0.011806 0.748375 0.0; 0.000343 0.570743 0.0 0.0 0.0 0.923967]), ((8, 2), [-0.22631 0.0 0.0 0.0 0.0 -0.000726; -4.6e-5 -0.076099 0.0 0.0 0.0 0.233771; 0.0 0.0 -0.11391 0.000153 -0.169443 0.0; 0.0 0.0 0.0 -0.656862 0.0 0.0; 0.0 0.0 0.402036 -0.011806 -0.748375 0.0; -0.000343 -0.570743 0.0 0.0 0.0 -0.923967]), ((3, 3), [0.825227 0.0 0.0 0.0 0.0 0.0; 0.0 0.44833 0.0 0.0 0.0 0.0; 0.0 0.0 0.659656 0.0 0.0 0.0; 0.0 0.0 0.0 1.21013 0.0 0.0; 0.0 0.0 0.0 0.0 1.438737 0.0; 0.0 0.0 0.0 0.0 0.0 1.732339]), ((5, 3), [0.16965 0.0 0.0 0.0 0.0 0.346352; 0.0 -0.22305 0.0 0.0 0.0 0.0; 0.0 0.0 0.025882 -0.241861 -0.0 -0.0; 0.0 0.0 0.44369 -0.604652 -0.0 -0.0; 0.0 0.0 0.0 -0.0 -0.719021 -0.0; -0.727072 0.0 0.0 0.0 0.0 -0.865881]), ((6, 3), [-0.412008 0.0 0.0 0.0 0.0 0.346352; 0.145414 -0.22305 0.0 0.0 0.0 -0.173176; 0.0 0.0 -0.32907 -0.241861 0.143804 0.0; 0.0 0.0 -0.44369 -0.604652 0.0 0.0; 0.0 0.0 0.0 0.0 -0.719021 0.0; 0.727072 0.0 0.0 0.0 0.0 -0.865881]), ((7, 3), [-0.412008 0.0 0.0 0.0 0.0 -0.346352; -0.145414 -0.22305 0.0 0.0 0.0 -0.173176; 0.0 0.0 -0.32907 0.241861 0.143804 0.0; 0.0 0.0 0.44369 -0.604652 -0.0 -0.0; 0.0 0.0 0.0 -0.0 -0.719021 -0.0; -0.727072 0.0 0.0 0.0 0.0 -0.865881]), ((8, 3), [-0.121179 0.0 0.0 0.0 0.0 0.0; -0.145414 -0.22305 0.0 0.0 0.0 -0.173176; 0.0 0.0 -0.151594 0.0 0.143804 0.0; 0.0 0.0 0.44369 -0.604652 -0.0 -0.0; 0.0 0.0 0.0 -0.0 -0.719021 -0.0; -0.727072 0.0 0.0 0.0 0.0 -0.865881]), ((4, 4), [0.66368 0.0 0.0 0.0 0.0 0.0; -0.038154 0.401187 0.0 0.0 0.0 0.0; 0.0 0.0 0.571519 0.0 0.0 0.0; 0.0 0.0 0.304869 0.969537 0.0 0.0; 0.0 0.0 0.116287 -0.044565 1.154391 0.0; -0.71199 -0.225565 0.0 0.0 0.0 1.124156]), ((5, 4), [0.046459 -0.063919 0.0 0.0 0.0 0.108625; 0.031117 -0.041468 0.0 0.0 0.0 0.07165; 0.0 0.0 -0.007206 0.004288 -0.010426 0.0; 0.0 0.0 0.020379 -0.024724 0.017422 0.0; 0.0 0.0 -0.028527 0.02418 -0.047083 0.0; 0.109977 -0.152667 0.0 0.0 0.0 0.256305]), ((6, 4), [0.146249 -0.063117 0.0 0.0 0.0 0.193352; -0.065952 0.184634 0.0 0.0 0.0 -0.118085; 0.0 0.0 0.213209 -0.18666 0.09026 0.0; 0.0 0.0 -0.207245 0.375492 0.014159 0.0; 0.0 0.0 0.116287 -0.032655 0.259116 0.0; 0.182986 -0.157854 0.0 0.0 0.0 0.369357]), ((7, 4), [0.512295 0.0 0.0 0.0 -0.0 -0.209269; 0.19514 0.24926 0.0 0.0 -0.0 -0.093259; 0.0 0.0 0.379818 0.188926 0.140965 0.0; 0.0 0.0 0.512114 0.591293 -0.051588 0.0; 0.0 0.0 0.0 0.041191 0.896129 0.0; -0.904051 0.0 0.0 0.0 0.0 0.761749]), ((8, 4), [-0.073055 -2.4e-5 0.0 0.0 0.0 -0.045609; 0.065952 -0.184634 0.0 0.0 0.0 0.118085; 0.0 0.0 -0.130311 0.036463 -0.095924 0.0; 0.0 0.0 0.207245 -0.375492 -0.014159 -0.0; 0.0 0.0 -0.116287 0.032655 -0.259116 0.0; -0.182986 0.157854 0.0 0.0 0.0 -0.369357]), ((5, 5), [1.401174 0.0 0.0 0.0 0.0 0.0; -0.126868 1.411994 0.0 0.0 0.0 0.0; 0.0 0.0 1.485455 0.0 0.0 0.0; 0.0 0.0 0.086459 1.498755 0.0 0.0; 0.0 0.0 0.053256 -0.089508 1.673372 0.0; -0.621371 -0.372499 0.0 0.0 0.0 0.16553]), ((6, 5), [-0.034623 0.000858 0.0 0.0 0.0 -0.104609; 0.007936 -0.043745 0.0 0.0 0.0 -0.604004; 0.0 0.0 -0.035945 0.008954 0.05517 0.0; 0.0 0.0 -0.091076 -0.110591 0.004404 0.0; 0.0 0.0 -0.006189 -0.00969 -0.333473 0.0; 0.159794 0.020567 0.0 0.0 0.0 -0.053123]), ((7, 5), [0.034759 -0.000671 0.0 0.0 0.0 0.115898; -0.009175 0.043339 0.0 0.0 0.0 0.575497; 0.0 0.0 0.035906 -0.00935 -0.055281 0.0; 0.0 0.0 0.091268 0.110799 -0.00417 0.0; 0.0 0.0 0.006171 0.009834 0.333673 0.0; -0.15692 -0.018731 0.0 0.0 0.0 0.128784]), ((8, 5), [0.098451 0.007368 0.0 0.0 0.0 0.087138; -0.007933 0.044097 0.0 0.0 0.0 0.605413; 0.0 0.0 0.072403 0.034961 -0.056992 0.0; 0.0 0.0 0.09121 0.110916 -0.004409 0.0; 0.0 0.0 0.006189 0.009708 0.333772 0.0; -0.159937 -0.020567 0.0 0.0 0.0 0.055607]), ((6, 6), [0.857719 0.0 0.0 0.0 0.0 0.0; -0.081221 0.627469 0.0 0.0 0.0 0.0; 0.0 0.0 0.886904 0.0 0.0 0.0; 0.0 0.0 -0.010247 0.962034 0.0 0.0; 0.0 0.0 -0.011866 0.025622 1.1389 0.0; 0.021852 -0.046161 0.0 0.0 0.0 0.988312]), ((7, 6), [-0.07081 0.125432 0.0 0.0 0.0 -0.008939; 0.113194 0.393343 0.0 0.0 0.0 0.041174; 0.0 0.0 -0.140359 0.007127 0.007819 0.0; 0.0 0.0 0.009158 -0.272995 -0.023286 0.0; 0.0 0.0 0.013448 -0.014056 -0.558683 0.0; 0.001926 0.102501 0.0 0.0 0.0 -0.303602]), ((8, 6), [-0.068435 0.094722 0.0 0.0 0.0 0.115022; 0.120869 0.417377 0.0 0.0 0.0 0.048094; 0.0 0.0 -0.137034 -0.101211 -0.000996 0.0; 0.0 0.0 0.010475 -0.273985 -0.021639 0.0; 0.0 0.0 0.011734 -0.010742 -0.558723 0.0; -0.016908 0.051542 0.0 0.0 0.0 -0.318871]), ((7, 7), [0.722947 0.0 0.0 0.0 0.0 0.0; -0.147567 0.245185 0.0 0.0 0.0 0.0; 0.0 0.0 0.761211 0.0 0.0 0.0; 0.0 0.0 -0.007567 0.815253 0.0 0.0; 0.0 0.0 -0.007927 0.005621 0.892877 0.0; -0.002527 -0.454236 0.0 0.0 0.0 0.668982]), ((8, 7), [-0.008694 -0.372879 0.0 0.0 0.0 -0.3873; -0.52446 -1.74135 0.0 0.0 0.0 -1.323073; 0.0 0.0 0.135329 0.08757 0.000138 0.0; 0.0 0.0 -0.006707 0.230911 0.013258 0.0; 0.0 0.0 -0.005992 0.010415 0.36325 0.0; -0.018569 -0.128761 0.0 0.0 0.0 0.215709]), ((8, 8), [2.364278 0.0 0.0 0.0 0.0 0.0; -0.534055 0.003076 0.0 0.0 0.0 0.0; 0.0 0.0 2.429332 0.0 0.0 0.0; 0.0 0.0 0.02072 2.435478 0.0 0.0; 0.0 0.0 -0.000314 0.000968 2.447206 0.0; -0.017766 1.790812 0.0 0.0 0.0 1.655084])]

b = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-10.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-10.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-10.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-10.0, 0.0, 0.0, 0.0, 0.0, 0.0]] 

RBD.cholsol(Lx, b)
==#