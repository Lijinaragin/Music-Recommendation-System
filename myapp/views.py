import datetime
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from django.http.response import JsonResponse
from django.shortcuts import render
from myapp.models import *
# Create your views here.
def login(request):
    return render(request,'login.html')
def loginPost(request):
    uname=request.POST["textfield"]
    pwd=request.POST["textfield2"]
    ob=logintbl.objects.filter(username=uname,password=pwd)
    if ob.exists():
        p=logintbl.objects.get(username=uname,password=pwd)
        if p.type =='admin':
            return HttpResponse('''<script>alert('Login');window.location='home'</script>''')
        else:
            return HttpResponse('''<script>alert('Invalid user name or password');window.location='/'</script>''')
    else:
        return HttpResponse('''<script>alert('Invalid user ');window.location='/'</script>''')


def home(request):
    return render(request,'index.html')

def user(request):
    ob = usertbl.objects.all()
    return render(request,'userview.html',{'data':ob})
def user_search(request):
    name=request.POST['textfield']
    ob=usertbl.objects.filter(name__istartswith=name)
    return render(request,'userview.html',{'data':ob,"n":name})

def song(request):
    ob = songtbl.objects.all()
    return render(request,'songview.html',{'data':ob})

def song_search(request):
    name=request.POST['textfield']
    ob=songtbl.objects.filter(songname__istartswith=name)
    return render(request,'songview.html',{'data':ob,"n":name})

def book(request):
    ob=booktbl.objects.all()
    return render(request,'BookView.html',{'data':ob})

def book_search(request):
    name=request.POST['textfield']
    ob=booktbl.objects.filter(bookname__istartswith=name)
    return render(request,'BookView.html',{'data':ob,"n":name})

# def book_ratings(request, book_id):
#     book = booktbl.objects.get(id=book_id)
#     ratings = bookrating.objects.filter(book=book)
#     return render(request, 'book_ratings.html', {'ratings': ratings, 'book': book})

def rest(request):
    ob = resttbl.objects.all()
    return render(request,'restview.html',{'data':ob})

def rest_search(request):
    name=request.POST['textfield']
    ob=resttbl.objects.filter(restname__istartswith=name)
    return render(request,'restview.html',{'data':ob,"n":name})

def feedback(request):
    ob=feedbacktbl.objects.all()
    return render(request,'feedbackview.html',{'data':ob})

def feedback_search(request):
    date=request.POST['textfield']
    ob=feedbacktbl.objects.filter(date__icontains=date)
    return render(request,'feedbackview.html',{'data':ob,"n":date})

def addsong(request):
    return render(request,'addsong.html')

def addsongPost(request):
    file = request.FILES["file"]
    s=FileSystemStorage()
    fn=s.save(file.name,file)
    ob=songtbl()
    ob.songname=request.POST["textfield"]
    ob.file=fn
    ob.save()
    return HttpResponse('''<script>alert('ADDED SUCCESSFULLY ');window.location='/song'</script>''')

def addbook(request):
    return render(request,'addbook.html')

def addbookPost(request):
    ob=booktbl()
    ob.bookname=request.POST["textfield"]
    ob.category=request.POST["textfield2"]
    ob.genre = request.POST["textfield3"]
    ob.author = request.POST["textfield4"]
    ob.ebooklink = request.POST["textfield5"]
    ob.save()
    return HttpResponse('''<script>alert('ADDED SUCCESSFULLY ');window.location='/book'</script>''')


def addrest(request):

    return render(request,'addrest.html')

def addrestPost(request):
    ob=resttbl()
    ob.restname=request.POST["textfield"]
    ob.lattitude=request.POST["textfield2"]
    ob.logitude = request.POST["textfield3"]
    ob.category = request.POST["cat"]
    ob.save()
    return HttpResponse('''<script>alert('ADDED SUCCESSFULLY ');window.location='/rest'</script>''')

def deletebook(request,id):
    ob=booktbl.objects.get(id=id)
    ob.delete()
    return HttpResponse('''<script>alert('deleted SUCCESSFULLY ');window.location='/book'</script>''')
def editbook(request,id):
    request.session["id"]=id
    ob=booktbl.objects.get(id=id)
    return render(request,'editbook.html',{'data':ob})
def editbookpost(request):
    ob = booktbl.objects.get(id=request.session["id"])
    ob.bookname = request.POST["textfield"]
    ob.category = request.POST["textfield2"]
    ob.genre = request.POST["textfield3"]
    ob.author = request.POST["textfield4"]
    ob.ebooklink = request.POST["textfield5"]
    ob.save()
    return HttpResponse('''<script>alert('updated SUCCESSFULLY ');window.location='/book'</script>''')

def deletesong(request,id):
    ob=songtbl.objects.get(id=id)
    ob.delete()
    return HttpResponse('''<script>alert('deleted SUCCESSFULLY ');window.location='/song'</script>''')
def editsong(request,id):
    request.session["id"]=id
    ob=songtbl.objects.get(id=id)
    return render(request,'editsong.html',{'data':ob})
def editsongpost(request):
    songname = request.POST["textfield"]
    ob = songtbl.objects.get(id=request.session["id"])
    if 'file' in request.FILES:
        file = request.FILES["file"]
        s = FileSystemStorage()
        fn = s.save(file.name, file)
        ob.file = fn

    ob.songname = songname
    ob.save()
    return HttpResponse('''<script>alert('updated SUCCESSFULLY ');window.location='/song'</script>''')

def deleterest(request,id):
    ob=resttbl.objects.get(id=id)
    ob.delete()
    return HttpResponse('''<script>alert('deleted SUCCESSFULLY ');window.location='/rest'</script>''')
def editrest(request,id):
    request.session["id"]=id
    ob=resttbl.objects.get(id=id)
    return render(request,'editrest.html',{'data':ob})
def editrestpost(request):
    restname = request.POST["textfield"]
    lattitude = request.POST["textfield2"]
    logitude = request.POST["textfield3"]
    category = request.POST["textfield4"]
    ob = resttbl.objects.get(id=request.session["id"])
    ob.restname = restname
    ob.lattitude = lattitude
    ob.logitude = logitude
    ob.category = category
    ob.save()
    return HttpResponse('''<script>alert('updated SUCCESSFULLY ');window.location='/rest'</script>''')





# -------------------android------------------------


def and_loginPost(request):
    uname=request.POST["username"]
    pwd=request.POST["password"]
    ob=logintbl.objects.filter(username=uname,password=pwd)
    if ob.exists():
        p=logintbl.objects.get(username=uname,password=pwd)
        if p.type =='user':
            return JsonResponse({"task":"valid","lid":p.id})
        else:
            return JsonResponse({"task": "invalid"})
    else:
        return JsonResponse({"task": "invalid"})


def and_register(request):
    ob=logintbl()
    ob.username=request.POST["username"]
    ob.password=request.POST["password"]
    ob.type='user'
    ob.save()
    p=usertbl()
    p.LOGIN=ob
    p.name=request.POST["name"]
    p.email=request.POST["email"]
    p.phone=request.POST["phone"]
    p.save()
    return  JsonResponse({'task':'valid'})


def viewsongs(request):
    list=[]
    ob=songtbl.objects.all()
    for i in ob:
        list.append({'id':i.id,'songname':i.songname,'file':i.file.url[1:]})
    print(list)
    return JsonResponse({'status':'ok','data':list})


def and_playlist(request):
    ob = songtbl()
    ob.Song_id= request.POST["songid"]
    ob.date=datetime.now()
    ob.save()
from .prediction import predict_img
def process_image(request):
    list=[]
    image=request.FILES.get("image")
    fs=FileSystemStorage()
    fn=fs.save(image.name,image)
    res=predict_img(r"C:\Users\Lenovo\PycharmProjects\music\media/"+fn)
    ob = songtbl.objects.filter(type=res)
    for i in ob:
        list.append({'id': i.id, 'songname': i.songname, 'file': i.file.url[1:]})
    print(list)
    return JsonResponse({'status': 'ok', 'data': res})

def playlistemo(request):
    list=[]
    res = request.POST['emo']
    print(res)
    ob = songtbl.objects.filter(type=res)
    for i in ob:
        list.append({'id': i.id, 'songname': i.songname, 'file': i.file.url[1:]})
    print(list)
    return JsonResponse({'status': 'ok', 'data': list})



def viewrest(request):
    list=[]
    ob=resttbl.objects.all()
    for i in ob:
        list.append({'id':i.id,'restname':i.restname,'category':i.category,'lattitude':i.lattitude,'logitude':i.logitude })
    print(list)
    return JsonResponse({'status':'ok','data':list})

def viewbook(request):
    list=[]
    ob=booktbl.objects.all()
    for i in ob:
        list.append({'id':i.id,'bookname':i.bookname,'category':i.category,'genre':i.genre,'author':i.author,'ebooklink':i.ebooklink })
    print(list)
    return JsonResponse({'status':'ok','data':list})


def usersendfeedback(request):
    feed = request.POST['feedback']
    lid = request.POST['lid']
    lob = feedbacktbl()
    lob.USER = usertbl.objects.get(LOGIN_id=lid)
    lob.feedback = feed
    lob.date = datetime.datetime.today()
    lob.save()
    return JsonResponse({'task': 'ok'})

def usersendrating(request):
    feed = request.POST['feedback']
    rating_value = request.POST['rating']
    lid = request.POST['lid']
    book_id = request.POST['book_id']

    lob = bookrating()
    lob.USER = usertbl.objects.get(LOGIN__id=lid)
    lob.book = booktbl.objects.get(id=book_id)
    lob.rating = rating_value
    lob.feedback = feed
    lob.date = datetime.datetime.today()
    lob.save()

    return JsonResponse({'task': 'ok'})