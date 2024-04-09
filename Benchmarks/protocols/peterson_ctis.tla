---- MODULE peterson_ctis ----
EXTENDS TLC, Naturals, FiniteSets
        
VARIABLE flag, turn, pc

Init == 
\/ flag = (0 :> FALSE @@ 1 :> TRUE) /\ pc = (0 :> "cs" @@ 1 :> "a1") /\ turn = 0
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a3" @@ 1 :> "a2") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a4" @@ 1 :> "a2") /\ turn = 0
\/ flag = (0 :> TRUE @@ 1 :> FALSE) /\ pc = (0 :> "a1" @@ 1 :> "a3") /\ turn = 0
\/ flag = (0 :> FALSE @@ 1 :> TRUE) /\ pc = (0 :> "a3" @@ 1 :> "a2") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> TRUE) /\ pc = (0 :> "a4" @@ 1 :> "a2") /\ turn = 0
\/ flag = (0 :> FALSE @@ 1 :> TRUE) /\ pc = (0 :> "a4" @@ 1 :> "a5") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a1" @@ 1 :> "a4") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> TRUE) /\ pc = (0 :> "a4" @@ 1 :> "a1") /\ turn = 0
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a4" @@ 1 :> "a1") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "cs" @@ 1 :> "a1") /\ turn = 0
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a2" @@ 1 :> "a3") /\ turn = 0
\/ flag = (0 :> TRUE @@ 1 :> FALSE) /\ pc = (0 :> "cs" @@ 1 :> "a4") /\ turn = 1
\/ flag = (0 :> TRUE @@ 1 :> FALSE) /\ pc = (0 :> "a3" @@ 1 :> "a3") /\ turn = 1
\/ flag = (0 :> TRUE @@ 1 :> FALSE) /\ pc = (0 :> "a3" @@ 1 :> "cs") /\ turn = 0
\/ flag = (0 :> FALSE @@ 1 :> TRUE) /\ pc = (0 :> "cs" @@ 1 :> "a4") /\ turn = 0
\/ flag = (0 :> FALSE @@ 1 :> TRUE) /\ pc = (0 :> "a3" @@ 1 :> "a1") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a3" @@ 1 :> "a4") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a1" @@ 1 :> "a3") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a3" @@ 1 :> "a3") /\ turn = 0
\/ flag = (0 :> FALSE @@ 1 :> TRUE) /\ pc = (0 :> "cs" @@ 1 :> "a3") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a3" @@ 1 :> "cs") /\ turn = 0
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "cs" @@ 1 :> "a2") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a3" @@ 1 :> "a4") /\ turn = 0
\/ flag = (0 :> FALSE @@ 1 :> TRUE) /\ pc = (0 :> "a3" @@ 1 :> "a1") /\ turn = 0
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "cs" @@ 1 :> "a4") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a1" @@ 1 :> "cs") /\ turn = 1
\/ flag = (0 :> TRUE @@ 1 :> TRUE) /\ pc = (0 :> "cs" @@ 1 :> "a4") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> TRUE) /\ pc = (0 :> "cs" @@ 1 :> "a5") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a5" @@ 1 :> "cs") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a4" @@ 1 :> "a3") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "cs" @@ 1 :> "a3") /\ turn = 0
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a5" @@ 1 :> "a4") /\ turn = 0
\/ flag = (0 :> TRUE @@ 1 :> FALSE) /\ pc = (0 :> "a5" @@ 1 :> "cs") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> TRUE) /\ pc = (0 :> "cs" @@ 1 :> "a2") /\ turn = 1
\/ flag = (0 :> TRUE @@ 1 :> FALSE) /\ pc = (0 :> "a5" @@ 1 :> "a4") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> TRUE) /\ pc = (0 :> "cs" @@ 1 :> "a4") /\ turn = 1
\/ flag = (0 :> TRUE @@ 1 :> FALSE) /\ pc = (0 :> "a2" @@ 1 :> "a4") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> TRUE) /\ pc = (0 :> "cs" @@ 1 :> "a5") /\ turn = 0
\/ flag = (0 :> TRUE @@ 1 :> FALSE) /\ pc = (0 :> "a3" @@ 1 :> "a4") /\ turn = 0
\/ flag = (0 :> TRUE @@ 1 :> FALSE) /\ pc = (0 :> "a2" @@ 1 :> "a4") /\ turn = 0
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a1" @@ 1 :> "a4") /\ turn = 0
\/ flag = (0 :> FALSE @@ 1 :> TRUE) /\ pc = (0 :> "cs" @@ 1 :> "a2") /\ turn = 0
\/ flag = (0 :> TRUE @@ 1 :> FALSE) /\ pc = (0 :> "a1" @@ 1 :> "a3") /\ turn = 1
\/ flag = (0 :> TRUE @@ 1 :> FALSE) /\ pc = (0 :> "a1" @@ 1 :> "a4") /\ turn = 0
\/ flag = (0 :> FALSE @@ 1 :> TRUE) /\ pc = (0 :> "a4" @@ 1 :> "a1") /\ turn = 1
\/ flag = (0 :> TRUE @@ 1 :> FALSE) /\ pc = (0 :> "a1" @@ 1 :> "a4") /\ turn = 1
\/ flag = (0 :> TRUE @@ 1 :> FALSE) /\ pc = (0 :> "a4" @@ 1 :> "cs") /\ turn = 0
\/ flag = (0 :> TRUE @@ 1 :> FALSE) /\ pc = (0 :> "a4" @@ 1 :> "cs") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> TRUE) /\ pc = (0 :> "a4" @@ 1 :> "a3") /\ turn = 0
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a3" @@ 1 :> "a1") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "cs" @@ 1 :> "a5") /\ turn = 0
\/ flag = (0 :> TRUE @@ 1 :> FALSE) /\ pc = (0 :> "a3" @@ 1 :> "a3") /\ turn = 0
\/ flag = (0 :> TRUE @@ 1 :> FALSE) /\ pc = (0 :> "a2" @@ 1 :> "a3") /\ turn = 0
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a4" @@ 1 :> "a4") /\ turn = 0
\/ flag = (0 :> TRUE @@ 1 :> FALSE) /\ pc = (0 :> "a2" @@ 1 :> "cs") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a3" @@ 1 :> "a2") /\ turn = 0
\/ flag = (0 :> TRUE @@ 1 :> FALSE) /\ pc = (0 :> "a1" @@ 1 :> "cs") /\ turn = 0
\/ flag = (0 :> FALSE @@ 1 :> TRUE) /\ pc = (0 :> "cs" @@ 1 :> "a1") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a2" @@ 1 :> "a4") /\ turn = 1
\/ flag = (0 :> TRUE @@ 1 :> FALSE) /\ pc = (0 :> "a1" @@ 1 :> "cs") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a2" @@ 1 :> "a4") /\ turn = 0
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a3" @@ 1 :> "cs") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a4" @@ 1 :> "a3") /\ turn = 0
\/ flag = (0 :> TRUE @@ 1 :> FALSE) /\ pc = (0 :> "a2" @@ 1 :> "cs") /\ turn = 0
\/ flag = (0 :> TRUE @@ 1 :> FALSE) /\ pc = (0 :> "a2" @@ 1 :> "a3") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a4" @@ 1 :> "a5") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a4" @@ 1 :> "a2") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> TRUE) /\ pc = (0 :> "a3" @@ 1 :> "a2") /\ turn = 0
\/ flag = (0 :> FALSE @@ 1 :> TRUE) /\ pc = (0 :> "cs" @@ 1 :> "a3") /\ turn = 0
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a4" @@ 1 :> "cs") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "cs" @@ 1 :> "a4") /\ turn = 0
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a4" @@ 1 :> "a4") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> TRUE) /\ pc = (0 :> "a3" @@ 1 :> "a3") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a5" @@ 1 :> "cs") /\ turn = 0
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "cs" @@ 1 :> "a1") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> TRUE) /\ pc = (0 :> "a4" @@ 1 :> "a2") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a2" @@ 1 :> "cs") /\ turn = 1
\/ flag = (0 :> TRUE @@ 1 :> FALSE) /\ pc = (0 :> "a4" @@ 1 :> "a4") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a4" @@ 1 :> "a1") /\ turn = 0
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a2" @@ 1 :> "cs") /\ turn = 0
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a3" @@ 1 :> "a3") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a4" @@ 1 :> "cs") /\ turn = 0
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a5" @@ 1 :> "a4") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> TRUE) /\ pc = (0 :> "a4" @@ 1 :> "a4") /\ turn = 0
\/ flag = (0 :> FALSE @@ 1 :> TRUE) /\ pc = (0 :> "a4" @@ 1 :> "a3") /\ turn = 1
\/ flag = (0 :> TRUE @@ 1 :> FALSE) /\ pc = (0 :> "a5" @@ 1 :> "a4") /\ turn = 0
\/ flag = (0 :> FALSE @@ 1 :> TRUE) /\ pc = (0 :> "a4" @@ 1 :> "a5") /\ turn = 0
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a2" @@ 1 :> "a3") /\ turn = 1
\/ flag = (0 :> TRUE @@ 1 :> FALSE) /\ pc = (0 :> "a3" @@ 1 :> "cs") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "cs" @@ 1 :> "a3") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "cs" @@ 1 :> "a2") /\ turn = 0
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a3" @@ 1 :> "a1") /\ turn = 0
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a4" @@ 1 :> "a5") /\ turn = 0
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "cs" @@ 1 :> "a5") /\ turn = 1
\/ flag = (0 :> FALSE @@ 1 :> TRUE) /\ pc = (0 :> "a3" @@ 1 :> "a3") /\ turn = 0
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a1" @@ 1 :> "cs") /\ turn = 0
\/ flag = (0 :> FALSE @@ 1 :> TRUE) /\ pc = (0 :> "a4" @@ 1 :> "cs") /\ turn = 0
\/ flag = (0 :> TRUE @@ 1 :> FALSE) /\ pc = (0 :> "a3" @@ 1 :> "a4") /\ turn = 1
\/ flag = (0 :> TRUE @@ 1 :> TRUE) /\ pc = (0 :> "a4" @@ 1 :> "cs") /\ turn = 0
\/ flag = (0 :> FALSE @@ 1 :> FALSE) /\ pc = (0 :> "a1" @@ 1 :> "a3") /\ turn = 0

Next == UNCHANGED <<flag,turn,pc>>

====