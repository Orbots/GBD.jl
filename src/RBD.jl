module RBD

export 
SolverBody, SolverConstraint3DofLinear, SolverConstraint1Dof, SolverConstraint1DofAngular,
solve, projectToManifoldVelocity, projectToR3Velocity!, 
setupPointToPointConstraint, setup1DofConstraint, setup1DofAngularConstraint,
skew, translation, create_transform, transform, inertia_tensor,
Matrix3x3, Vector3, Matrix4x4, Transform, step_world, RigidBody, to_solver_body, 
PointToPointConstraint, Linear1DofConstraint, ContactConstraint, Angular1DofConstraint

using LinearAlgebra
using StaticArrays
using ReferenceFrameRotations

Scalar = Float64
Vector3 = SVector{3, Scalar}
Matrix3x3 = SMatrix{3,3, Scalar, 9}
Matrix4x4 = SMatrix{4,4, Scalar, 16}
Transform = Matrix4x4

#Base.:*(A::AT, v::VT) where{T, S, AT<:SMatrix{4,4,T,16}, VT<:SVector{3,S}} = SVector{3,S}((A*vcat(v, 1))[1:3])
transform(A::AT, v::VT) where{T, S, AT<:SMatrix{4,4,T,16}, VT<:SVector{3,S}} = SVector{3,S}((A*vcat(v, 1))[1:3])
transform(A::AT, v::VT) where{T, S, AT<:Matrix{T}, VT<:SVector{3,S}} = SVector{3,S}((A*vcat(v, 1))[1:3])
rotate(A::AT, v) where{T, AT<:SMatrix{4,4,T,16}} = SVector{3,T}(A[1:3,1:3]*v)
rotate(A::AT, v) where{T, AT<:Matrix{T}} = SVector{3,T}(A[1:3,1:3]*v)
translation(wTb) = Vector3(wTb[4,1:3])

create_transform(R, t) = vcat(hcat(R, zeros(3)), hcat(t', 1)) |> Transform

Base.:>>>(q::T, v) where {T<:Quaternion} = vect(q*v*inv(q))

skew(ω::Vector3) = [0.0 -ω[3] ω[2];
         ω[3] 0.0  -ω[1];
         -ω[2] ω[1] 0.0]

function inertia_tensor(box::Vector3, m::Scalar)
    x,y,z = box
    (m/12.0) * Vector3(y^2+z^2, x^2+z^2, x^2+y^2)
end

function generate_frame(z)
    y = Vector3(0,1,0)
    if abs(y⋅z) < 0.9
        x = normalize(y×z)
        y = z×x
    else
        x = Vector3(1,0,0)
        y = normalize(z×x)
        x = y×z
    end
    (x,y,z)
end

import LinearAlgebra.normalize

normalize(q::Quaternion) = q/norm(q)

##=========== Constraint Solver ==============##

mutable struct SolverBody
    v::Vector3
    ω::Vector3

    #Iinv::Vector3   # hmmm... if we want in localspace need transform stored
    #priciple_axis_transform::Quaternion   # transform into space where inertia tensor I is diagonal
    wTb::Transform

    Iinv::Matrix3x3   # in worldspace
    massInv::Scalar
end

abstract type SolverConstraint end
abstract type SolverConstraintClamped <: SolverConstraint end

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

mutable struct SolverConstraintContact <: SolverConstraintClamped
    normal_constraint::SolverConstraint1Dof
    friction_constraint_id::Int64
end

SolverConstraintContact(bodyA, bodyB) = SolverConstraintContact(
    SolverConstraint1Dof(0, 0, bodyA, bodyB, Vector3(zeros(3)), Vector3(zeros(3)), Vector3(zeros(3)), 0), 0)

unclamped_constraint(c:: SolverConstraintContact) = c.normal_constraint

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
    Matrix3x3(zeros(3,3)), Matrix3x3(zeros(3,3)), Matrix3x3(zeros(3,3)))

function setup1DofConstraint( dt_inv, wTa::Transform, wTb::Transform, pointOnAInA::Vector3, pointOnBInB::Vector3, worldAxis::Vector3, c::SolverConstraint1Dof )
    r1 = rotate(wTa, pointOnAInA);
    r2 = rotate(wTb, pointOnBInB);

    c.Ja = r1×worldAxis  # non-permissable angular axis
    c.Jb = -r2×worldAxis

    c.Dinv = inv((c.bodyA.Iinv*c.Ja)⋅c.Ja + (c.bodyB.Iinv*c.Jb)⋅c.Jb + c.bodyA.massInv + c.bodyB.massInv)
    c.axis = worldAxis

    x1 = translation(wTa) + r1;
    x2 = translation(wTb) + r2;

    # Compute the positional constraint error (scaled by the Baumgarte coefficient 'beta')
    positionError = (x1 - x2);
    scaleError = dt_inv*0.2;
    positionError = positionError*scaleError;

    c.manifoldVelocityToCorrect = positionError⋅c.axis;

    return c
end


function projectToManifoldVelocity( c::SolverConstraint1Dof )
    p = c.bodyA.v ⋅ c.axis;
    p -= c.bodyB.v ⋅ c.axis;

    p += c.Ja ⋅ c.bodyA.ω;
    p += c.Jb ⋅ c.bodyB.ω;

    return p;
end

function projectToR3Velocity!(  c::SolverConstraint1Dof, dp::Scalar )
    dv = dp * c.bodyA.massInv;
    c.bodyA.v += dv * c.axis;
    dv = dp * c.bodyB.massInv;
    c.bodyB.v -= dv * c.axis;
    dω = c.Ja*dp;
    dω = c.bodyA.Iinv*dω;
    c.bodyA.ω += dω;
    dω = c.Jb*dp;
    dω = c.bodyB.Iinv*dω;
    c.bodyB.ω += dω;
    c
end

"""
    provide two mutually perpendicular axis which are also perpendicular to the axis of the prohibited rotation.  One on body A and one on body B.
    this relation should be met when the constraint is at rest a⟂×b⟂ = constrained
"""
function setup1DofAngularConstraint( dt_inv, perp_axisa::Vector3, perp_axisb::Vector3, c::SolverConstraint1DofAngular )

    axis = normalize(perp_axisa×perp_axisb)
    c.Ja = axis
    c.Jb = -c.Ja

    c.Dinv = inv((c.bodyA.Iinv*c.Ja)⋅c.Ja + (c.bodyB.Iinv*c.Jb)⋅c.Jb)

    # Compute the constraint error (scaled by the Baumgarte coefficient 'beta')
    angularError = perp_axisa⋅perp_axisb;
    scaleError = dt_inv*0.2;
    angularError = angularError*scaleError;

    c.manifoldVelocityToCorrect = angularError;

    return c
end


function projectToManifoldVelocity( c::SolverConstraint1DofAngular )
    p = c.Ja ⋅ c.bodyA.ω;
    p += c.Jb ⋅ c.bodyB.ω;

    return p;
end

function projectToR3Velocity!(  c::SolverConstraint1DofAngular, dp::Scalar )
    dω = c.Ja*dp;
    dω = c.bodyA.Iinv*dω;
    c.bodyA.ω += dω;
    dω = c.Jb*dp;
    dω = c.bodyB.Iinv*dω;
    c.bodyB.ω += dω;
    c
end

function setupContactConstraint( dt_inv, wTa::Transform, wTb::Transform, pointOnAInA::Vector3, pointOnBInB::Vector3, worldAxis::Vector3, c::SolverConstraintContact )
    c.normal_constraint = setup1DofConstraint( dt_inv, wTa, wTb, pointOnAInA, pointOnBInB, worldAxis, c.normal_constraint )
    c
end

projectToManifoldVelocity( c::SolverConstraintContact ) = projectToManifoldVelocity(c.normal_constraint)

function projectToR3Velocity!(  c::SolverConstraintContact, dp::Scalar )
    c.normal_constraint = projectToR3Velocity!(c.normal_constraint, dp)
    c
end

function projectToManifoldVelocity( c::SolverConstraint3DofLinear )
    p = c.bodyA.v;
    p -= c.bodyB.v;

    ppart = c.bodyA.ω;
    p += c.angularJa*ppart;

    ppart = c.bodyB.ω;
    ppart = c.angularJb*ppart;
    p += ppart;

    return p;
end

function projectToR3Velocity!(  c::SolverConstraint3DofLinear, dp::Vector3 )
    dv = dp * c.bodyA.massInv;
    c.bodyA.v += dv;
    dv = dp * c.bodyB.massInv;
    c.bodyB.v -= dv;
    dω = dp;
    dω = c.angularJa'*dω;
    dω = c.bodyA.Iinv*dω;
    c.bodyA.ω += dω;
    dω = dp;
    dω = c.angularJb'*dω;
    dω = c.bodyB.Iinv*dω;
    c.bodyB.ω += dω;
    c
end

# output Dinv and solutionImpulse->zero
# to be more Julian should change to method returning Dinv
function init( c::SolverConstraint3DofLinear )
    Dinv = Matrix3x3(diagm(repeat([c.bodyA.massInv+c.bodyB.massInv], 3)))

    Jt = c.angularJa'
    dpartM = c.angularJa*c.bodyA.Iinv;
    dpartM *= Jt;
    Dinv += dpartM;

    Jt = c.angularJb'
    dpartM = c.angularJb*c.bodyB.Iinv;
    dpartM *= Jt;
    Dinv += dpartM;

    c.Dinv = Matrix3x3(inv(Dinv));

    c.solutionImpulse = zero(c.solutionImpulse);

    return c;
end


function setupPointToPointConstraint( dt_inv, wTa::Transform, wTb::Transform, pointOnAInA::Vector3, pointOnBInB::Vector3, c )
    r1 = rotate(wTa, pointOnAInA);
    r2 = rotate(wTb, pointOnBInB);

    x1 = translation(wTa) + r1;
    x2 = translation(wTb) + r2;

    r1 = -r1;
    c.angularJa = skew(r1)
    c.angularJb = skew(r2)

    # Compute the positional constraint error (scaled by the Baumgarte coefficient 'beta')
    positionError = (x1 - x2);
    scaleError = dt_inv*0.2;
    positionError = positionError*scaleError;

    c.manifoldVelocityToCorrect = positionError;

    return init(c)
end

function solve( c::SolverConstraint )
    # project velocities onto constraint manifold. i.e. this gives us the magnitude of the velocity violating the constraint conditions
    manV = projectToManifoldVelocity(c);

    manV += c.manifoldVelocityToCorrect;

    Δimpulse = -c.Dinv*manV;  # momentum now

    projectToR3Velocity!( c, Δimpulse );

    c.solutionImpulse += Δimpulse;
end

# clamps accumulated solution impulse to be > 0
function solve( cc::SolverConstraintClamped )
    c = unclamped_constraint(cc)
    # project velocities onto constraint manifold. i.e. this gives us the magnitude of the velocity violating the constraint conditions
    manV = projectToManifoldVelocity(c);

    manV += c.manifoldVelocityToCorrect;

    Δimpulse = -c.Dinv*manV
    solutionImpulse′ = max.(zero(Scalar), c.solutionImpulse + Δimpulse);
    Δimpulse = solutionImpulse′ - c.solutionImpulse 

    projectToR3Velocity!( c, Δimpulse );

    c.solutionImpulse += Δimpulse;
end

##=========== Stepper ==============##

mutable struct RigidBody
    x::Vector3
    m::Scalar
    I::Vector3   # bodyspace
    v::Vector3
    ω::Vector3   # world space
    q::Quaternion{Scalar}
end

RigidBody(x,m=1.0,inertia=ones(3)) = RigidBody(x, m, inertia, 
                                                   zeros(3), zeros(3), Quaternion(1.0,0,0,0))

apply_inertia(q::Quaternion, I, v) = q>>>(I.*inv(q)>>>v)

function ω_implicit(ω₀::Vector3, It, q::Quaternion, dt)
    # one step of newtons method to solve for new angular velocity = f(ω′) = I(ω′-ω)+ω′xIω′*dt = 0
    # df(ω′)/ω′ = I + (1xIω′+ω′xI)*dt
    # df(ω) = I + (ωxI - Iωx1)*dt
    ω = inv(q)>>>ω₀
    Iω = It*ω
    f = ω×Iω*dt 
    df = It + (skew(ω)*It - skew(Iω))*dt
   
    ω′ = ω - df\f
    q>>>ω′
end

function integrate_implicit!(rb::RigidBody, dt::Scalar)
    rb.x += rb.v*dt
    rb.ω = ω_implicit(rb.ω, Diagonal(rb.I), rb.q, dt)
    q2 = rb.q + 0.5*dt*rb.ω*rb.q
    rb.q = q2/norm(q2)
    
    return rb
end

function step_world(world, dt::Scalar)
    for rb in world
        integrate_implicit!(rb, dt)
    end
end

function step_world(bodies, constraints, dt::Scalar; external_force = rb->zero(Vector3), external_torque = rb->zero(Vector3))
    # Symplectic Euler.  update velocity before constraints and position after constraints
    # integrate velocity
    for rb in bodies
        rb.v += external_force(rb)*dt
        rb.ω += external_torque(rb)*dt
        rb.ω = ω_implicit(rb.ω, Diagonal(rb.I), rb.q, dt)
    end

    solver_bodies = map(to_solver_body, bodies)

    constraints = map(cᵢ->to_solver_constraint(cᵢ, solver_bodies, dt), constraints)

    # solve constraints for updated velocity
    for cᵢ in constraints
        solve(cᵢ)
    end

    for (i, bodᵢ) in enumerate(solver_bodies)
        bodies[i].v = bodᵢ.v
        bodies[i].ω = bodᵢ.ω
    end

    # integrate positions with updated velocities
    for rb in bodies
        rb.x += rb.v*dt
        rb.q = normalize(rb.q + 0.5*dt*rb.ω*rb.q)
    end
end

#===========  Entities ==============#

struct PointToPointConstraint
    bodyA::Int64
    bodyB::Int64
    ra::Vector3  # offset from com to pivot in space of body A
    rb::Vector3
end

struct Linear1DofConstraint
    bodyA::Int64
    bodyB::Int64
    ra::Vector3  # offset from com to pivot in space of body A
    rb::Vector3
    axisb::Vector3   # axis of constraint in body B space
end

struct Angular1DofConstraint
    bodyA::Int64
    bodyB::Int64
    perp_axisa::Vector3  # offset from com to pivot in space of body A
    perp_axisb::Vector3
end

Angular1DofConstraint(bodyA, bodyB, axis::Vector3, qa::Quaternion, qb::Quaternion) = Angular1DofConstraint(bodyA, bodyB, ([inv(qa), inv(qb)].>>>generate_frame(axis)[1:2])...)

struct ContactConstraint
    bodyA::Int64
    bodyB::Int64
    ra::Vector3  # offset from com to pivot in space of body A
    rb::Vector3
    normal::Vector3   # contact normal in body B space
end

function to_solver_body(rb::RigidBody)
    R = quat_to_dcm(rb.q)' 
    Iinv_world = R*diagm(inv.(rb.I))*R' 

    wTb = create_transform(Array(quat_to_dcm(rb.q)'), rb.x)
    SolverBody(rb.v, rb.ω, wTb, Iinv_world, inv(rb.m))
end

function to_solver_constraint(p2p::PointToPointConstraint, solver_bodies, dt)
    rba = solver_bodies[p2p.bodyA]
    rbb = solver_bodies[p2p.bodyB]
    setupPointToPointConstraint( 1.0/dt, rba.wTb, rbb.wTb, p2p.ra, p2p.rb, SolverConstraint3DofLinear(rba, rbb) )
end

function to_solver_constraint(c::Linear1DofConstraint, solver_bodies, dt)
    rba = solver_bodies[c.bodyA]
    rbb = solver_bodies[c.bodyB]
    setup1DofConstraint( 1.0/dt, rba.wTb, rbb.wTb, c.ra, c.rb, rotate(rbb.wTb, c.axisb), SolverConstraint1Dof(rba, rbb) )
end

function to_solver_constraint(c::Angular1DofConstraint, solver_bodies, dt)
    rba = solver_bodies[c.bodyA]
    rbb = solver_bodies[c.bodyB]
    setup1DofAngularConstraint( 1.0/dt, rotate(rba.wTb, c.perp_axisa), rotate(rbb.wTb, c.perp_axisb), SolverConstraint1DofAngular(rba, rbb) )
end

function to_solver_constraint(c::ContactConstraint, solver_bodies, dt)
    rba = solver_bodies[c.bodyA]
    rbb = solver_bodies[c.bodyB]
    setupContactConstraint( 1.0/dt, rba.wTb, rbb.wTb, c.ra, c.rb, rotate(rbb.wTb, c.normal), SolverConstraintContact(rba, rbb) )
end

end # module

#==
using .RBD
using LinearAlgebra, StaticArrays, ReferenceFrameRotations

dt = 0.0167
box_radius = Vector3(0.05,0.2,0.4)
offset = Vector3(0,0.0,0)
x = offset
rb = RigidBody(x)
rb.q = Quaternion(1.0,0,0,0); rb.ω = [0.0,0,8]; rb.x = x; rb.v = [0.0,0.0,-1.0]
rb.m = 10
rb.I = inertia_tensor(Vector3(2*box_radius), rb.m)

x2 = offset+[1.0,0,0].*(box_radius*2)
rb2 = RigidBody(x2)
rb2.q = Quaternion(1.0,0,0,0); rb2.ω = [0.0,0,8]; rb2.x = x2; rb2.v = [0.0,0.0,-1.0]
rb2.m = 10
rb2.I = inertia_tensor(Vector3(2*box_radius), rb2.m)

p2p = PointToPointConstraint(1, 2, Vector3([1,1,1].*(box_radius)), Vector3([-1,1,1].*(box_radius)))
constraints = [p2p]

c2 = Angular1DofConstraint(1, 2, Vector3(1,0,0), rb.q, rb2.q)
#c3 = ContactConstraint(1, 2, Vector3([-1,1,-1].*(box_radius)), Vector3([1,1,1].*(box_radius)), normalize(Vector3(0,0,-1)))
constraints = vcat(constraints, [c1, c2])


plane_radius = Vector3(5,5,0.5)
rb3 = RigidBody(Vector3(0,0,-3))
rb3.q = normalize(Quaternion(1.0,0.2,0,0)); rb3.ω = [0.0,0,0]; rb3.v = [0.0,0.0,-0.0]
rb3.m = 1000000
rb3.I = inertia_tensor(Vector3(plane_radius*2), rb3.m)

c1 = ContactConstraint(1, 3, Vector3([1,-1,-1].*(box_radius)), Vector3([0,0,1].*(plane_radius)), normalize(Vector3(0,0,1)))
c2 = ContactConstraint(1, 3, Vector3([-1,1,-1].*(box_radius)), Vector3([0,0,1].*(plane_radius)), normalize(Vector3(0,0,1)))
c3 = ContactConstraint(1, 3, Vector3([-1,-1,1].*(box_radius)), Vector3([0,0,1].*(plane_radius)), normalize(Vector3(0,0,1)))
c4 = ContactConstraint(1, 3, Vector3([-1,-1,-1].*(box_radius)), Vector3([0,0,1].*(plane_radius)), normalize(Vector3(0,0,1)))
c5 = ContactConstraint(1, 3, Vector3([-1,1,1].*(box_radius)), Vector3([0,0,1].*(plane_radius)), normalize(Vector3(0,0,1)))
c6 = ContactConstraint(1, 3, Vector3([1,-1,1].*(box_radius)), Vector3([0,0,1].*(plane_radius)), normalize(Vector3(0,0,1)))
c7 = ContactConstraint(1, 3, Vector3([1,1,-1].*(box_radius)), Vector3([0,0,1].*(plane_radius)), normalize(Vector3(0,0,1)))
c8 = ContactConstraint(1, 3, Vector3([1,1,1].*(box_radius)), Vector3([0,0,1].*(plane_radius)), normalize(Vector3(0,0,1)))
constraints = vcat(constraints, [c1, c2, c3, c4, c5, c6, c7, c8])
c1 = ContactConstraint(2, 3, Vector3([1,-1,-1].*(box_radius)), Vector3([0,0,1].*(plane_radius)), normalize(Vector3(0,0,1)))
c2 = ContactConstraint(2, 3, Vector3([-1,1,-1].*(box_radius)), Vector3([0,0,1].*(plane_radius)), normalize(Vector3(0,0,1)))
c3 = ContactConstraint(2, 3, Vector3([-1,-1,1].*(box_radius)), Vector3([0,0,1].*(plane_radius)), normalize(Vector3(0,0,1)))
c4 = ContactConstraint(2, 3, Vector3([-1,-1,-1].*(box_radius)), Vector3([0,0,1].*(plane_radius)), normalize(Vector3(0,0,1)))
c5 = ContactConstraint(2, 3, Vector3([-1,1,1].*(box_radius)), Vector3([0,0,1].*(plane_radius)), normalize(Vector3(0,0,1)))
c6 = ContactConstraint(2, 3, Vector3([1,-1,1].*(box_radius)), Vector3([0,0,1].*(plane_radius)), normalize(Vector3(0,0,1)))
c7 = ContactConstraint(2, 3, Vector3([1,1,-1].*(box_radius)), Vector3([0,0,1].*(plane_radius)), normalize(Vector3(0,0,1)))
c8 = ContactConstraint(2, 3, Vector3([1,1,1].*(box_radius)), Vector3([0,0,1].*(plane_radius)), normalize(Vector3(0,0,1)))
constraints = vcat(constraints, [c1, c2, c3, c4, c5, c6, c7, c8])



niter = 10

drawit = true
# DRAW
using MeshCat, GeometryBasics, CoordinateTransformations
if drawit
    vis = Visualizer(); anim = Animation()
    setobject!(vis[:box1], Rect(-Vec(box_radius), Vec(box_radius)*2)); 
    setobject!(vis[:box2], Rect(-Vec(box_radius), Vec(box_radius)*2));
    setobject!(vis[:plane], Rect(-Vec(plane_radius), Vec(plane_radius)*2));
    atframe(anim, 0) do
        settransform!(vis[:box1], compose(Translation(rb.x...), LinearMap(quat_to_dcm(rb.q)')))
        settransform!(vis[:box2], compose(Translation(rb2.x...), LinearMap(quat_to_dcm(rb2.q)')))
        settransform!(vis[:plane], compose(Translation(rb3.x...), LinearMap(quat_to_dcm(rb3.q)')))
    end
end
#

bodies = [rb, rb2, rb3]
g = Vector3(0,0,-9.81)
for i in 1:400
    
    rb.v += g*dt
    rb2.v += g*dt
    step_world(bodies, constraints, dt)

    # DRAW
    if drawit
        atframe(anim, i) do
            settransform!(vis[:box1], compose(Translation(rb.x...), LinearMap(quat_to_dcm(rb.q)')))
            settransform!(vis[:box2], compose(Translation(rb2.x...), LinearMap(quat_to_dcm(rb2.q)')))
            settransform!(vis[:plane], compose(Translation(rb3.x...), LinearMap(quat_to_dcm(rb3.q)')))
        end
    end
end


# DRAW
if drawit
    setanimation!(vis, anim)
end

==#

sleep(3)
