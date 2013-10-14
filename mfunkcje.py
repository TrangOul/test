import math,numpy,copy,time,os,random

'''
AX=B
LUX=B

{LY=B
{UX=Y
'''

def ludoolittle(A):
	n=A.shape[0] #wymiar macierzy (formalnie liczba wierszy, ale mamy macierz kwadratowa)
	L=numpy.eye(n) #macierz L, poczatkowo identycznosciowa
	U=numpy.zeros((n,n)) #macierz U, poczatkowo z zerami
	for i in range(n): #dla kazdego wiersza
		#wypelnia macierze L i U naprzemian
		for j in range(i,n): #dla kazdego elementu na przekatnej i powyzej
			t=A[i,j]
			for k in range(i): t-=L[i,k]*U[k,j]
			U[i,j]=t
		for j in range(i+1,n): #dla kazdego elementu ponizej przekatnej 
			t=A[j,i]
			for k in range(i): t-=L[j,k]*U[k,i]
			try: L[j,i]=t/U[i,i]
			except ZeroDivisionError: return None
	return L,U

def rozwiazlu(A,B): #ludoolittle tylko rozklada macierz A na L i U, a ta funkcja rozwiazuje uklad AX=LUX=B
	L,U=ludoolittle(A)
	Y=podstawianieWsteczned(L,B)
	X=podstawianieWsteczneg(U,Y)
	return X

def uklad_z_pliku(nazwapliku):
	' wczytuje z pliku macierz zapisaną wierszami oraz wektor wyrazów wolnych'
	with open(nazwapliku) as f:
		dane = f.readlines()
		r = [ ] #lista na wyrazy wolny
		for i in range(len(dane)):
			dane[i] = dane[i].split() #podziel po bialych znakach
			dane[i] = [ numpy.float(x) for x in dane[i]] #zamien na liczby
			r.append(numpy.float(dane[i][-1])) #dopisz wyraz wolny
			dane[i] = dane[i][0 : -1] #zamien na wiersz macierzy
	return numpy.matrix(dane), numpy.array(r)

def cramer(A,B):
	n=A.shape[0]
	X=numpy.zeros(n)
	detA=numpy.linalg.det(A)
	if detA==0: return None
	for i in range(n):
		A2=copy.deepcopy(A) # moze wystarczy shallow copy?
		A2[:,i]=numpy.transpose(numpy.matrix(B)) #zamieniamy n-ta kolumne macierzy A na kolumne wyrazow wolnych
		X[i]=numpy.linalg.det(A2)/detA
	return X

def odwracanie(A,B): return numpy.dot(numpy.linalg.inv(A),B)

def generujmacierz(n1,n2=None,zeros=False,dom=False):
	m=20
	if n2==None: n2=n1
	if n1!=n2: dom=False #macierz niekwadratowa nie ma przekatnej
	A=numpy.zeros((n1,n2))
	if dom:
		for i in range(n1):
			for j in range(n2):	
				if i!=j and zeros: A[i,j]=random.randint(-m,m)
				elif i!=j and not zeros: A[i,j]=random.choice([-1,1])*random.randint(1,m)
			A[i,i]=random.choice([-1,1])*(numpy.sum(numpy.absolute(A[i,]))+random.randint(1,m))
	else:
		for i in range(n1):
			if zeros: A[i]=numpy.array([random.randint(-m,m) for k in list(range(n2))])
			else: A[i]=numpy.array([random.choice([-1,1])*random.randint(1,m) for k in list(range(n2))])
	return A

def generujmacierz12(n): return numpy.matrix(numpy.ones((n,n))+numpy.eye(n))

def generujmacierzrzadka(n,k=0): #generuje macierz z losowa liczba niezerowych elementow poza przekatna i elementem dominujacym na przekatnej
	m=20
	if k>n*n-n: return None
	A=numpy.zeros((n,n),dtype=numpy.float)
	przekatna=[i*n+i for i in range(n)] #lista numerow elementow na przekatnej. Elementy unikalne, mozna zamienic na zbior
	#nieprzekatna=[x for x in range(n*n) if x not in przekatna] #wolno
	#nieprzekatna=list(range(n*n))              # wolno
	#for i in przekatna: nieprzekatna.remove(i) #
	nieprzekatna=set(range(n*n))-set(przekatna) #najszybciej
	losowe=random.sample(nieprzekatna,k)
	for i in losowe: A.flat[i]=random.choice([-1,1])*random.randint(1,m)
	for i in range(n): A.flat[przekatna[i]]=random.choice([-1,1])*(numpy.sum(numpy.absolute(A[i,]))+random.randint(1,m)) #element dominujacy na przekatnej - warunek metody Gaussa-Seidla
	return A

def generujmacierzwstegowa(n,k1,k2=None,dom=False): #generuje macierz wstegowa o szerokosci k1 albo k1+k2+1
	m=20
	if k2==None: 
		k1=k1//2 #jak nie podamy grubosci wstegi nad i pod przekatna, sam wyliczy z calkowitej grubosci
		k2=k1
	A=numpy.zeros((n,n),dtype=numpy.float) #macierz, ktora wypelnimy
	for i in range(n): #dla kazdego wiersza
		if dom:
			for j in range(n): #dla kazdej kolumny
				if j>=i-k1 and j<=i+k2 and i!=j: A[i,j]=random.choice([-1,1])*random.randint(1,m) #jak element jest na wstedze, ale nie na przekatnej, wpisz losowa liczbe +-(1-m)
			A[i,i]=random.choice([-1,1])*(numpy.sum(numpy.absolute(A[i,]))+random.randint(1,m)) #na przekatnej wpisz liczbe wieksza od sumy elementow w wierszu (dominacja przekatnej - warunek zbieznosci Gaussa-Seidla)
		else:
			for j in range(n): #dla kazdej kolumny
				if j>=i-k1 and j<=i+k2: A[i,j]=random.choice([-1,1])*random.randint(1,m) #jak element jest na wstedze, wpisz losowa liczbe +-(1-m)
	return A


def podstawianieWsteczneg(m, r):
	'rozwiązuje układ równań m x = r z macierzą trójkątną górną'
	n = len(r)
	x = numpy.zeros(n)
	for i in range(n-1,-1,-1): #dla kazdego wiersza od konca do poczatku (od ostatniego (n-1, bo numrujemy od 0) do -1 (czyli 0, bo -1 wylacznie), z krokiem -1
		s = r[i]
		for j in range(i+1,n):
			s -= m[i,j]*x[j] #odejmij od wyrazu wolnego zmienne poza ta, ktora teraz liczymy
		x[i] = s / m[i,i] #podziel przez wspolczynnik
	return x

def podstawianieWsteczned(m, r):
	'rozwiązuje układ równań m x = r z macierzą trójkątną dolną'
	n = len(r)
	x = numpy.zeros(n)
	for i in range(0,n,1): #dla kazdego wiersza od poczatku do konca
		s = r[i]
		for j in range(0,i):
			s -= m[i,j]*x[j]
		x[i] = s / m[i,i]
	return x

def gauss0(m, r):
	'''
		rozwiązuje równanie m x = r metodą eliminacji
		metoda naiwna, bez sprawdzania zer na przekątnej
		niepoprawna algorytmicznie; może nastąpić próba dzielenia przez zero
	'''
	n = len(r)
	assert m.shape == (n,n)
	for i in range(n):
		baza = m[i,i]
		if baza == 0.0:
			# czy oznacza to, że nie ma rozwiązania?
			return None
		for j in range(i+1,n):
			wsp = m[j, i] / baza
			m[j, i:n] -= m[i, i:n] * wsp
			r[j] -= r[i] * wsp
	return podstawianieWsteczneg(m, r)


def gauss1(m, r):
	'''
		rozwiązuje równanie m x = r metodą eliminacji
		metoda z wymianą wierszy w przypadku zera na przekątnej
		niepoprawna numerycznie; wyniki nie są wiarygodne
	'''
	n = len(r)
	assert m.shape == (n,n)
	prawieZero = 1.0e-15
	for i in range(n):
		baza = m[i, i]
		if abs(baza) < prawieZero:
			for j in range(i+1, n):
				if abs(m[j,i]) >= prawieZero:
					t = copy(m[i])
					m[i] = m[j]
					m[j] = t
					r[i], r[j] = r[j], r[i]
					baza = m[i, i]
					break
		if abs(baza) < prawieZero:
			# co to oznacza?
			return None
		for j in range(i+1,n):
			wsp = m[j, i] / baza
			m[j, i:n] -= m[i, i:n] * wsp
			r[j] -= r[i] * wsp
	return podstawianieWsteczneg(m, r)


def gauss2(m, r):
	'''
		rozwiązuje równanie m x = r metodą eliminacji
		metoda z częściowym wyborem elementu głównego (o największym module)
		jest on poszukiwany w kolumnie roboczej
	'''
	n = len(r)
	assert m.shape == (n,n) #sprawdzenie, czy macierz jest kwadratowa
	prawieZero = 1.0e-15
	for i in range(n): #dla kazdej kolumny
		baza = abs(m[i, i]) #jako baze zapisz element na przekatnej
		ibaza = i           #zapisz tez indeks
		for j in range(i+1, n): #dla kazdego wiersza ponizej przekatnej ('podkolumny' od elementu ponizej przekatnej do konca w dol)
			t = abs(m[j,i])
			if t > baza: #jezeli modul elementu w podkolumnie jest > od elementu na przekatnej
				baza = t #zapisz element
				ibaza = j #zapisz indeks
		if i != ibaza: #jezeli znaleziono element wiekszy co do modulu od tego na przekatnej
			t = copy.copy(m[i]) #zamien wiersze 
			m[i] = m[ibaza]     #nie mozna bezposrednio a,b=b,a, trzeba zapisac do zmiennej tymczasowej:
			m[ibaza] = t        #t=a; a=b; b=t
			r[i], r[ibaza] = r[ibaza], r[i] #zamien odpowiadajace wyrazy wolne
		if baza < prawieZero:
			# co to oznacza? jak nie znaleziono elementu roznego od (prawie) zera, ukladu nie da sie rozwiazac ta metoda
			return None
		baza = m[i,i]
		for j in range(i+1,n): #tu juz metoda eliminacji Gaussa; dla kazdego elementu w 'podkolumnie'
			wsp = m[j, i] / baza
			m[j, i:n] -= m[i, i:n] * wsp
			r[j] -= r[i] * wsp
	return podstawianieWsteczneg(m, r)

def gauss_2(m, r):
	n = len(r)
	assert m.shape == (n,n)
	prawieZero = 1.0e-15
	for i in range(n-1):
		baza = m[i, i]
		maxcol=numpy.amax(numpy.absolute(m[i+1:,i])) #element najwiekszy co do modulu w 'podkolumnie'
		maxcolind=numpy.argmax(numpy.absolute(m[i+1:,i])) #indeks powyzszego elementu
		if maxcol > abs(baza): #jezeli znaleziono wiekszy co do modulu od przekatnej
			t = copy.copy(m[i])      #zamiana wierszy w macierzy wspolczynnikow
			m[i] = m[i+1+maxcolind]  #
			m[i+1+maxcolind] = t     #
			r[i], r[i+1+maxcolind] = r[i+1+maxcolind], r[i] #zamiana wierszy w macierzy wyrazow wolnych
			baza = m[i, i]
		if abs(baza) < prawieZero:
			# co to oznacza?
			return None
		for j in range(i+1,n):
			wsp = m[j, i] / baza
			m[j, i:n] -= m[i, i:n] * wsp
			r[j] -= r[i] * wsp
	return podstawianieWsteczneg(m, r)


def gauss_seidel(m, r, x = None, maxiter = 1000, eps = 1e-6):
	'''
		rozwiązuje równanie m x = r metodą iteracyjną Gaussa-Seidla
		maxiter: dopuszczalna liczba iteracji
		eps: żądana dokładność rozwiązania
		x: przybliżenie startowe (jeśli nie podano, przyjmowany jest wektor zerowy)
	'''
	n = len(r)
	assert m.shape == (n,n)
	for i in range(n):
		if m[i,i] == 0.0: return None #jak 0 na przekatnej, nie do rozwiazania ta metoda
	if x == None: x = numpy.zeros(n) #jak nie wybrano przyblizenia poczatkowego, przyjmij zera
	i = 0
	while i < maxiter: #maksymalnie dla zadanej liczby iteracji wykonaj
		i=i+1
		blad = 0.0
		for j in range(n): #dla kazdego wyrazu wolnego = dla kazdego wiersza
			t = r[j] #wyraz wolny
			for k in range(n): #dla kazdej kolumny
				if  k != j: t -= m[j, k]*x[k] #jezeli nie na przekatnej, odejmij od w. wolnego iloczyny elementu wiersza i odpowiedniej zmiennej (przyblizonej)
			t = t / m[j, j] #podziel przez wyraz na przekatnej
			blad = max(blad, abs(x[j] - t)) #blad to roznica poprzedniego przyblizenia i wyliczonego w tej iteracji
			x[j] = t #zapisz przyblizenie zmiennej
		if blad < eps: return x #kiedy roznica przyblizen jest < od zadanej tj. kolejne iteracje nie poprawiaja znaczaco wyniku, zakoncz
	return None #jezeli po zadanej liczbie iteracji blad bedzie duzy, to znaczy, ze dla danej macierzy metoda nie jest zbiezna


def gauss_seidelmap(m, r, mapa=None, x = None, maxiter = 1000, eps = 1e-6):
	'''
		rozwiązuje równanie m x = r metodą iteracyjną Gaussa-Seidla
		maxiter: dopuszczalna liczba iteracji
		eps: żądana dokładność rozwiązania
		x: przybliżenie startowe (jeśli nie podano, przyjmowany jest wektor zerowy)
	'''
	n = len(r)
	assert m.shape == (n,n)
	if mapa==None: mapa=mapabp(m)
	for i in range(n):
		if m[i,i] == 0.0:
			return None
	if x == None:
		x = numpy.zeros(n)
	i = 0
	while i < maxiter:
		i=i+1
		blad = 0.0
		for j in range(n):
			t = r[j]
			for k in mapa[j]: #nie dla kazdego elementu wiersza, tylko dla tych z mapy
				t -= m[j, k]*x[k] #nie trzeba sprawdzac, przekatna usunieta przez mape
			t = t / m[j, j]
			blad = max(blad, abs(x[j] - t))
			x[j] = t
		if blad < eps: return x
	return None


#mapa wierszy (lista list z indexami niezerowych elementow)
def mapa(A):
	m=[]
	w,k=A.shape
	for i in range(w):
		m1=[]
		for j in range(k):
			if A[i,j]!=0: m1.append(j) #dokladnie 0, bo operujemy na danych wejsciowych
		m.append(copy.copy(m1))
	return m

def mapabp(A):
	n=A.shape[0]
	assert A.shape == (n,n)
	m=[] #lista na listy niezerowych i nieprzekatnych elementow
	for i in range(n): #dla kazdego wiersza
		m1=[] #lista na nie0 i nieprz elementy
		for j in range(n):
			if A[i,j]!=0 and i!=j: m1.append(j) #zapisz, jezeli !=0 i nie na przekatnej; dokladnie 0, nie prawie, bo operujemy na danych wejsciowych
		m.append(copy.copy(m1)) #dopisz liste
	return m

def przekdom(A,B): #zamiana na macierz przekatniowo dominujaca
	n=len(B)
	assert A.shape == (n,n)
	for i in range(n-1):
		maxcol=numpy.amax(numpy.absolute(A[i+1:,i]))
		maxcolind=numpy.argmax(numpy.absolute(A[i+1:,i]))
		if maxcol > abs(A[i,i]):
			t = copy.copy(A[i])      #zamiana wierszy w macierzy wspolczynnikow
			A[i] = A[i+1+maxcolind]  #
			A[i+1+maxcolind] = t     #
			B[i], B[i+1+maxcolind] = B[i+1+maxcolind], B[i] #zamiana wierszy w macierzy wyrazow wolnych
	return A,B

def normalizuj(A,B): #normalizuje macierz tak, ze na przekatnej sa 1
	n=len(B)
	assert A.shape == (n,n)
	for i in range(n):
		baza=A[i,i]
		A[i]/=baza
		B[i]/=baza
	return A,B

'''
def mnk1(x,y):
	#assert len(x)==len(y)
	A=numpy.zeros((2,2))
	B=numpy.zeros(2)
	for i in range(len(x)):
		x2=(x[i])**2
		A[0][0]+=x2
		A[0][1]+=x[i]
		B[0]+=x[i]*y[i]
		B[1]+=y[i]
	A[1][0]=A[0][1]
	A[1][1]=len(x)
	return A,B
'''

#metoda najmniejszych kwadratow stopnia k dla 1 zmiennej niezaleznej
def mnk(x,y,k=1):
	assert len(x)==len(y) and k<=len(x)
	A=numpy.zeros((k+1,k+1))
	B=numpy.zeros(k+1)
	potegix=numpy.zeros(2*k+1)
	#sumy=0.0
	for i in range(len(x)):
		potegix[0]=1 #unikamy 0**0 da xi=0
		for j in range(1,2*k+1):
			potegix[j]=potegix[j-1]*x[i]
		for j in range(k+1):
			for m in range(k+1):
				A[j,m]+=potegix[j+m]
			B[j]+=y[i]*potegix[j]
	return A,B


#metoda najmniejszych kwadratow stopnia 1 dla 2 zmiennych niezaleznych
def mnk2(x,y): #x jako krotka z 2 listami punktow
	assert len(x[0])==len(y) and len(x[1])==len(y)
	A=numpy.zeros((3,3))
	B=numpy.zeros(3)
	potegix=[]
	potegix.append(numpy.zeros(3))
	potegix.append(numpy.zeros(3))
	#sumy=0.0
	for i in range(len(x[0])):
		potegix[0][0]=1
		potegix[1][0]=1
		for j in range(1,3):
			potegix[0][j]=potegix[0][j-1]*x[0][i]
			potegix[1][j]=potegix[1][j-1]*x[1][i]
		for j in range(3):
			for m in range(3):
				if   j+m==0: A[j,m]+=1
				elif j+m==1: A[j,m]+=potegix[0][1]
				elif j+m==2 and j!=m: A[j,m]+=potegix[1][1]
				elif j+m==2 and j==m: A[j,m]+=potegix[0][2]
				elif j+m==4 and j==m: A[j,m]+=potegix[1][2]
				elif j+m==3: A[j,m]+=(potegix[0][1]*potegix[1][1])
			if j==0: B[j]+=y[i]
			else:
				B[j]+=y[i]*potegix[j-1][1]
	return A,B


def ustawprio(p=2):
	import win32api,win32process,win32con
	#priorityclasses=[win32process.IDLE_PRIORITY_CLASS,        #64
    #                win32process.BELOW_NORMAL_PRIORITY_CLASS, #16384
    #                win32process.NORMAL_PRIORITY_CLASS,       #32
    #                win32process.ABOVE_NORMAL_PRIORITY_CLASS, #32768
    #                win32process.HIGH_PRIORITY_CLASS,         #128
    #                win32process.REALTIME_PRIORITY_CLASS]     #256  #5 nie dziala, ustawia 4
	priorityclasses=[64,16384,32,32768,128,256]
	pid=win32api.GetCurrentProcessId()
	win32process.SetPriorityClass(win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS,True,pid),priorityclasses[p]) #ustaw priorytet
	prio={64:0, 16384:1, 32:2, 32768:3, 128:4, 256:5}[win32process.GetPriorityClass(win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS,True,pid))] #pobierz priorytet i zamien na numer zgodny z p
	if prio!=p: print('Procesowi '+str(pid)+' nie mozna ustawic priorytetu '+str(p)+', ustawiono '+str(prio)+'.') #ostrzezenie w razie niepowodzenia (w zasadzie tylko 4 zamiast 5)

def wezprio():
	prio={64:'niski (0)', 16384:'ponizej normalnego (1)', 32:'normalny (2)', 32768:'powyzej normalnego (3)', 128:'wysoki (4)', 256:'czasu rzeczywistego (5)'}
	import win32api,win32process,win32con
	pid=win32api.GetCurrentProcessId()
	p=win32process.GetPriorityClass(win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS,True,pid))
	print('Proces '+str(pid)+' ma priorytet '+prio[p]+'.')
	return p
