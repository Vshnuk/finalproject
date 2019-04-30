<?php
    
    session_start();
    if(isset($_POST['query_submit']))
    {
        if (isset($_FILES['image']['name']) && $_FILES['image']['error']== 0)
        {

            
            $tmpFilePath = $_FILES["image"]["tmp_name"];
            $newFilePath = "uploads/". $_FILES["image"]["name"];
            foreach (new DirectoryIterator('/var/www/html/majorproject/uploads') as $fileInfo) {
                if(!$fileInfo->isDot()) {
                    unlink($fileInfo->getPathname());
                }
            }
            $re=move_uploaded_file($tmpFilePath, $newFilePath);
            if($re)
            {
                echo "successful";
            }
            else
            {  
            echo "Error !! , Your image not uploaded.";
            }
       }
       else
       {
           echo "unsuccessful";
       }
    }
    else if(isset($_POST['query_run']))
    {
        echo shell_exec('python /var/www/html/majorproject/search.py');
        header("Location: result.php");
    }
    else
    {
        $_SESSION['login_successful']=$_SESSION['login_successful']?$_SESSION['login_successful']:false;
        if(!$_SESSION['login_successful'])
            header("Location: login.html");
    }

?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Document</title>

    <link href="//maxcdn.bootstrapcdn.com/bootstrap/4.1.1/css/bootstrap.min.css" rel="stylesheet" id="bootstrap-css">
<script src="//maxcdn.bootstrapcdn.com/bootstrap/4.1.1/js/bootstrap.min.js"></script>
<script src="//cdnjs.cloudflare.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
</head>
<body>
    <br><br>
    <div class="container">
        <div class="row">
            <div class="col-sm-10"></div>
            <div class="col-sm-2"><a href="logout.php"><u>Logout</u></a></div>
        </div>
        <br><br><br>
        <div class="row">
            <div class="col-sm-3"></div>
            <div class="col-sm-6"><h1><u>Upload query image</u></h1></div>
        </div>
        <br><br><br>
        <form id="fileUploadForm"  action=""  enctype="multipart/form-data" method="POST">
            <fieldset>
                <div class="form-horizontal">
                    <div class="form-group">
                        <div class="row">
                        <label class="control-label col-md-2 text-right" for="filename"><span>File</span></label>
                        <div class="col-md-12">
                            <div class="input-group">
                                <input type="hidden" id="filename" name="filename" value="">
                                <input type="file" id="uploadedFile" name="image" class="form-control form-control-sm" accept=".csv, application/vnd.openxmlformats-officedocument.spreadsheetml.sheet, application/vnd.ms-excel">
                                <div class="input-group-btn">
                                    <input type="submit" value="Upload" class="rounded-0 btn btn-primary" name="query_submit">
                                    <input type="submit" value="Search" class="rounded-0 btn btn-primary" name="query_run">
                                </div>
                            </div>
                        </div>
                        </div>
                    </div>                        
                </div>
            </fieldset>    
        </form>
    </div>
</body>
</html>