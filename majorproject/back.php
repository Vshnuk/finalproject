<?php

session_start();
$_SESSION['login_successful']=$_SESSION['login_successful']?$_SESSION['login_successful']:false;
if(!$_SESSION['login_successful'])
    header("Location: login.html");
foreach (new DirectoryIterator('/var/www/html/majorproject/resultimages') as $fileInfo) {
    if(!$fileInfo->isDot()) {
        unlink($fileInfo->getPathname());
    }
}
header("Location: user.php");
?>